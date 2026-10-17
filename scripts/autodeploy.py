"""Local CI-gated deployment agent. Only trusted main is deployed; no dataset leaves the Mac."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import logging
import logging.handlers
import os
import plistlib
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import venv
from ipaddress import ip_address, ip_network
from pathlib import Path

REPO = "madagascar-ai/forte-bank-madagascar"
LABEL = "kz.forte-guard.autodeploy"
DEFAULT_ROOT = Path.home() / "Library" / "Application Support" / "ForteGuard"
SHA = re.compile(r"^[0-9a-f]{40}$")
LOG = logging.getLogger("forte-deploy")


def team_llm_host(hostname):
    # Mirrors forte_guard.settings.is_team_llm_host; this agent runs without the package installed.
    if hostname in {"localhost", "ollama"}:
        return True
    try:
        address = ip_address(hostname or "")
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address in ip_network("100.64.0.0/10")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)


def environment(root, release=None):
    values = os.environ.copy()
    for line in (root / ".env").read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            values[key] = value.strip().strip("\"'")
    # All persistent state belongs to this runtime, never a developer's checkout.
    values["FORTE_DATA_DIR"] = str(root / "data")
    model = Path(values.get("FORTE_MODEL_PATH", "models/detector-v2.pt"))
    values["FORTE_MODEL_PATH"] = str(model if model.is_absolute() else root / model)
    values["FORTE_DEPLOY_STATE"] = str(root / "status.json")
    values["PYTHONUNBUFFERED"] = "1"
    if release:
        values["PYTHONPATH"] = str(Path(release["source"]) / "src")
        values["FORTE_CONFIG_DIR"] = str(Path(release["source"]) / "config")
        values["FORTE_RELEASE_SHA"] = release["sha"]
    return values


def request_json(url, headers=None, data=None, timeout=8):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers=headers or {}, data=data)
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


class GitHub:
    def __init__(self):
        self.auth = None

    def get(self, path):
        if self.auth is None:
            result = subprocess.run(
                ["git", "credential", "fill"],
                input=f"protocol=https\nhost=github.com\npath={REPO}.git\n\n",
                text=True,
                capture_output=True,
                check=True,
                timeout=20,
            )
            fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            self.auth = base64.b64encode(
                f"{fields.get('username', '')}:{fields['password']}".encode()
            ).decode()
        try:
            return request_json(
                "https://api.github.com" + path,
                headers={
                    "Authorization": "Basic " + self.auth,
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "Forte-local-deployment",
                },
                timeout=20,
            )
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                self.auth = None
            raise

    def head(self):
        sha = self.get(f"/repos/{REPO}/git/ref/heads/main")["object"]["sha"]
        if not SHA.fullmatch(sha):
            raise ValueError("Invalid commit ID")
        return sha

    def ci(self, sha):
        query = urllib.parse.urlencode({"head_sha": sha, "branch": "main", "event": "push", "per_page": 20})
        runs = self.get(f"/repos/{REPO}/actions/workflows/tests.yml/runs?{query}")["workflow_runs"]
        eligible = [
            run
            for run in runs
            if run["head_sha"] == sha
            and run["head_branch"] == "main"
            and run["event"] == "push"
            and run["repository"]["full_name"] == REPO
        ]
        # A newer failed rerun must supersede an earlier success.
        return max(eligible, key=lambda run: (run["id"], run.get("run_attempt", 1))) if eligible else None


def set_maintenance(root, enabled):
    database = root / "data" / "jobs.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database, timeout=15) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("CREATE TABLE IF NOT EXISTS service_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO service_state VALUES ('maintenance', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("1" if enabled else "0",),
        )


def active_jobs(root):
    with sqlite3.connect(root / "data" / "jobs.sqlite3", timeout=15) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='jobs'").fetchone()
        return (
            conn.execute("SELECT count(*) FROM jobs WHERE status='processing'").fetchone()[0] if exists else 0
        )


def activate(candidate, previous, *, maintenance, drain, stop, start, ready, save):
    """Drain first; publish the new pointer only after readiness. Restore code AND environment on failure."""
    maintenance(True)
    changed = False
    try:
        drain()
        stop()
        changed = True
        start(candidate)
        if not ready(candidate):
            raise RuntimeError("CandidateReadinessFailed")
        save(candidate)
    except Exception:
        if changed:
            stop()
            if previous:
                start(previous)
                if not ready(previous):
                    raise RuntimeError("RollbackReadinessFailed") from None
                save(previous)
        raise
    finally:
        maintenance(False)


class Agent:
    def __init__(self, root, interval=30):
        self.root = Path(root).resolve()
        self.interval = interval
        self.github = GitHub()
        self.done = threading.Event()
        self.child = None
        self.ollama = None
        self.live = read_json(root / "live.json") or None
        self.status = read_json(root / "status.json")
        self.failed_sha, self.failed_at = None, 0

    def update(self, **values):
        self.status.update(values, checked_at=time.time())
        self.status["active_sha"] = self.live["sha"] if self.live else None
        write_json(self.root / "status.json", self.status)

    def url(self):
        return "http://127.0.0.1:" + environment(self.root).get("FORTE_PORT", "8000")

    def ready(self, release, seconds=45):
        deadline = time.monotonic() + seconds
        key = environment(self.root)["FORTE_API_KEY"]
        while time.monotonic() < deadline and not self.done.is_set():
            try:
                health = request_json(self.url() + "/health", timeout=3)
                readiness = request_json(self.url() + "/api/v1/readiness", {"X-API-Key": key}, timeout=4)
                if health.get("release") == release["sha"] and readiness.get("status") == "ready":
                    return True
            except (OSError, ValueError):
                pass
            if self.child is not None and self.child.poll() is not None:
                return False
            self.done.wait(1)
        return False

    def stop_api(self):
        if self.child is not None:
            if self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait(timeout=5)
            self.child = None
            return
        # Recover only a process whose recorded command has our runtime env and exact SHA.
        record = read_json(self.root / "api-process.json")
        pid, sha = record.get("pid"), record.get("sha")
        if not pid or not sha:
            return
        command = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
        ).stdout
        if command and (str(self.root / "envs") not in command or f"--deployment-id {sha}" not in command):
            raise RuntimeError("RecordedProcessDoesNotBelongToAgent")
        if command:
            os.kill(pid, signal.SIGTERM)
            for _ in range(30):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(1)
            else:
                raise RuntimeError("PreviousProcessDidNotStop")
        (self.root / "api-process.json").unlink(missing_ok=True)

    def start_api(self, release):
        log_path = self.root / "logs" / f"api-{release['sha'][:12]}.log"
        with log_path.open("ab") as output:
            self.child = subprocess.Popen(
                [release["python"], "-m", "forte_guard.cli", "serve", "--deployment-id", release["sha"]],
                cwd=self.root,
                env=environment(self.root, release),
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        write_json(self.root / "api-process.json", {"pid": self.child.pid, "sha": release["sha"]})

    def save(self, release):
        self.live = release
        write_json(self.root / "live.json", release)

    def drain(self):
        seconds = float(environment(self.root).get("FORTE_JOB_TIMEOUT_SEC", "285")) + 30
        deadline = time.monotonic() + seconds
        while active_jobs(self.root):
            if self.done.wait(1) or time.monotonic() > deadline:
                raise RuntimeError("ActiveJobsDidNotDrain")

    def prepare(self, sha):
        if not SHA.fullmatch(sha):
            raise ValueError("Invalid commit ID")
        mirror = self.root / "repository.git"
        if not mirror.exists():
            subprocess.run(["git", "init", "--bare", str(mirror)], capture_output=True, check=True)
            subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(mirror),
                    "remote",
                    "add",
                    "origin",
                    f"https://github.com/{REPO}.git",
                ],
                check=True,
            )
        subprocess.run(
            # "+" accepts a rewritten main (force-push); only a SHA with green CI is ever deployed.
            ["git", "--git-dir", str(mirror), "fetch", "origin", "+refs/heads/main:refs/heads/main"],
            capture_output=True,
            check=True,
            timeout=60,
        )
        source = self.root / "releases" / sha
        if not (source / ".exported").exists():
            source.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile() as archive:
                subprocess.run(["git", "--git-dir", str(mirror), "archive", sha], stdout=archive, check=True)
                archive.seek(0)
                with tarfile.open(fileobj=archive) as package:
                    package.extractall(source, filter="data")
            (source / ".exported").touch()
        digest = hashlib.sha256(sys.version.encode())
        for name in ("pyproject.toml", "constraints-runtime.txt"):
            digest.update((source / name).read_bytes())
        env_dir = self.root / "envs" / digest.hexdigest()[:20]
        if not (env_dir / ".ready").exists():
            if env_dir.exists():
                shutil.rmtree(env_dir)
            venv.EnvBuilder(with_pip=True).create(env_dir)
            with (self.root / "logs" / f"install-{env_dir.name}.log").open("ab") as output:
                subprocess.run(
                    [
                        str(env_dir / "bin/python"),
                        "-m",
                        "pip",
                        "install",
                        "--cache-dir",
                        str(self.root / "pip-cache"),
                        "-c",
                        str(source / "constraints-runtime.txt"),
                        f"{source}[vision]",
                    ],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=900,
                )
                subprocess.run(
                    [str(env_dir / "bin/python"), "-m", "pip", "check"],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=30,
                )
            (env_dir / ".ready").touch()
        release = {
            "sha": sha,
            "source": str(source),
            "python": str(env_dir / "bin/python"),
            "deployed_at": time.time(),
        }
        # Load the actual local detector once, and probe Ollama, before stopping the active API.
        probe = "from forte_guard.settings import Settings; from forte_guard.readiness import inspect_readiness; from forte_guard.vision.detector import YoloTracker; s=Settings(); s.validate(); assert inspect_readiness(s)['status']=='ready'; YoloTracker(s)"
        with (self.root / "logs" / f"preflight-{sha[:12]}.log").open("ab") as output:
            subprocess.run(
                [release["python"], "-c", probe],
                cwd=self.root,
                env=environment(self.root, release),
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=90,
            )
        return release

    def ensure_ollama(self):
        env = environment(self.root)
        if env.get("FORTE_LLM_ENABLED", "true").lower() not in {"1", "true", "yes"}:
            return
        url = env.get("FORTE_OLLAMA_URL", "http://127.0.0.1:11434")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "http" or not team_llm_host(parsed.hostname):
            raise ValueError("OllamaMustBeLocal")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            # A team GPU server is run by its owner; the release preflight still requires it to be ready.
            return
        try:
            request_json(url.rstrip("/") + "/api/tags", timeout=3)
            return
        except OSError:
            pass
        if self.ollama is None or self.ollama.poll() is not None:
            binary = shutil.which("ollama") or "/Applications/Ollama.app/Contents/Resources/ollama"
            with (self.root / "logs/ollama.log").open("ab") as output:
                self.ollama = subprocess.Popen(
                    [binary, "serve"],
                    env={**os.environ, "OLLAMA_HOST": parsed.netloc},
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
        raise RuntimeError("WaitingForOllama")

    def tick(self):
        self.ensure_ollama()
        if self.live and (self.child is None or self.child.poll() is not None):
            self.stop_api()
            self.start_api(self.live)
            if not self.ready(self.live):
                raise RuntimeError("ActiveReleaseRestartFailed")
            set_maintenance(self.root, False)
        sha = self.github.head()
        if self.live and sha == self.live["sha"]:
            self.update(state="ready", pending_sha=None, error=None)
            return
        run = self.github.ci(sha)
        if not run or run["status"] != "completed" or run["conclusion"] != "success":
            self.update(
                state="ci_failed" if run and run["status"] == "completed" else "waiting_for_ci",
                pending_sha=sha,
                ci_url=run.get("html_url") if run else None,
                error=None,
            )
            return
        if sha == self.failed_sha and time.time() - self.failed_at < 300:
            return
        self.update(state="preparing", pending_sha=sha, ci_url=run["html_url"], error=None)
        candidate = self.prepare(sha)
        if self.done.is_set() or self.github.head() != sha:
            return
        previous = self.live
        self.update(state="deploying", previous_sha=previous["sha"] if previous else None)
        activate(
            candidate,
            previous,
            maintenance=lambda value: set_maintenance(self.root, value),
            drain=self.drain,
            stop=self.stop_api,
            start=self.start_api,
            ready=self.ready,
            save=self.save,
        )
        self.update(state="ready", pending_sha=None, deployed_at=candidate["deployed_at"], error=None)
        LOG.info("Deployed %s", sha)

    def run(self):
        with (self.root / "agent.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_json(self.root / "agent-process.json", {"pid": os.getpid()})
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda *_: self.done.set())
            try:
                while not self.done.is_set():
                    try:
                        self.tick()
                    except Exception as exc:
                        self.failed_sha = self.status.get("pending_sha")
                        self.failed_at = time.time()
                        code = type(exc).__name__
                        if isinstance(exc, RuntimeError):
                            code = str(exc)
                        self.update(state="error_keeping_previous", error=code)
                        LOG.error("Deployment error: %s", code)
                    # Supervise the API between GitHub polls too.
                    for _ in range(max(1, int(self.interval / 2))):
                        if self.done.wait(2):
                            break
                        if self.live and self.child is not None and self.child.poll() is not None:
                            break
            finally:
                self.stop_api()
                set_maintenance(self.root, False)
                if self.ollama is not None and self.ollama.poll() is None:
                    self.ollama.terminate()


def bootstrap(root, workspace, sync_data=False):
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    for folder in ("logs", "data", "models", "releases", "envs"):
        (root / folder).mkdir(exist_ok=True)
    if not (root / ".env").exists():
        shutil.copyfile(workspace / ".env", root / ".env")
        (root / ".env").chmod(0o600)
    if not (root / ".models-copied").exists():
        shutil.copytree(workspace / "models", root / "models", dirs_exist_ok=True)
        (root / ".models-copied").touch()
    if sync_data:
        if (root / "live.json").exists():
            raise RuntimeError("RefusingToOverwriteManagedData")
        source = workspace / "data"
        shutil.copytree(source / "jobs", root / "data/jobs", dirs_exist_ok=True)
        with (
            sqlite3.connect(source / "jobs.sqlite3") as original,
            sqlite3.connect(root / "data/jobs.sqlite3") as target,
        ):
            original.backup(target)
    # Stable controller code lives outside the working checkout and release directories.
    if Path(__file__).resolve() != (root / "autodeploy.py").resolve():
        shutil.copyfile(__file__, root / "autodeploy.py")
    write_json(root / "controller.json", {"python": str(Path(sys.executable).resolve()), "root": str(root)})


def launch_agent(root, action):
    domain = f"gui/{os.getuid()}"
    plist = Path.home() / "Library/LaunchAgents" / (LABEL + ".plist")
    if action == "install":
        controller = read_json(root / "controller.json")
        plist.parent.mkdir(parents=True, exist_ok=True)
        settings = {
            "Label": LABEL,
            "ProgramArguments": [
                controller["python"],
                "-u",
                str(root / "autodeploy.py"),
                "--root",
                str(root),
                "run",
            ],
            "WorkingDirectory": str(root),
            "RunAtLoad": True,
            "KeepAlive": True,
            "ThrottleInterval": 30,
            "ExitTimeOut": 40,
            "EnvironmentVariables": {
                "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            },
            "StandardOutPath": str(root / "logs/launchd.log"),
            "StandardErrorPath": str(root / "logs/launchd.log"),
        }
        plist.write_bytes(plistlib.dumps(settings))
        plist.chmod(0o600)
        subprocess.run(["launchctl", "bootstrap", domain, str(plist)], check=True)
    elif action == "stop":
        subprocess.run(["launchctl", "bootout", domain + "/" + LABEL], check=True)
    elif action == "start":
        subprocess.run(["launchctl", "bootstrap", domain, str(plist)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("bootstrap")
    setup.add_argument("--workspace", type=Path, required=True)
    setup.add_argument("--sync-data", action="store_true")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("sha")
    commands.add_parser("run")
    commands.add_parser("status")
    for name in ("install", "start", "stop"):
        commands.add_parser(name)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "bootstrap":
        bootstrap(root, args.workspace.resolve(), args.sync_data)
        print("Deployment runtime prepared:", root)
    elif args.command in ("install", "start", "stop"):
        launch_agent(root, args.command)
    elif args.command == "status":
        print(json.dumps(read_json(root / "status.json"), ensure_ascii=False, indent=2))
    else:
        handler = logging.handlers.RotatingFileHandler(
            root / "logs/agent.log", maxBytes=5_000_000, backupCount=3
        )
        logging.basicConfig(
            level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)s %(message)s"
        )
        agent = Agent(root)
        if args.command == "prepare":
            agent.ensure_ollama()
            print(json.dumps(agent.prepare(args.sha)))
        else:
            agent.run()


if __name__ == "__main__":
    main()
