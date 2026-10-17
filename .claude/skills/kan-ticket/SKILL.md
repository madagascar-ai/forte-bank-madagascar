---
name: kan-ticket
description: Take a Jira KAN ticket for Forte Guard (a given key like KAN-12, or the next agent-ready one, optionally limited to a module label) and execute it end-to-end by the AGENTS.md protocol — claim, branch, implement, verify every acceptance criterion and the DoD, open a PR, report in Jira. Use when the user says "возьми тикет", "следующий тикет", "/kan-ticket".
---

# Взять и выполнить тикет KAN

Аргументы (необязательные): ключ тикета (`KAN-12`) и/или метка модуля (`vision`, `journal`, `reasoning`, `api`, `ui`, `eval`, `infra`, …). Пример: `/kan-ticket next journal`.

Источник правил — `AGENTS.md`, раздел 8 (протокол) и раздел 2.2 (данные). Этот skill только задаёт порядок шагов.

1. **Jira.** cloudId `6c9f5c38-f623-4d07-a3e6-2d1938d5f9fa`. Если инструментов `atlassian` нет, попроси участника выполнить `/mcp` и войти, либо вставить текст тикета.
2. **Выбор.** Если ключ дан — `getJiraIssue`. Иначе `searchJiraIssuesUsingJql` с запросом из AGENTS.md 8.3 (+ `AND labels = <модуль>`, если модуль указан). Пропусти тикеты, у которых есть связь «is blocked by» на незакрытый тикет.
3. **Взятие.** `atlassianUserInfo` → `editJiraIssue` (assignee = свой accountId) → `transitionJiraIssue` (id 21) → перечитай: исполнитель не ты — бери следующий. Комментарий о взятии (`addCommentToJiraIssue`) с веткой и планом.
4. **Подготовка.** `git switch main && git pull --ff-only && git switch -c kan-<N>-<slug>`. Прочитай тикет, связанные тикеты и код из «Текущего состояния»; расхождения — комментарием.
5. **Работа.** Только «Что сделать», ничего из «Вне рамок». Посторонняя работа — новый тикет (`createJiraIssue`, метки `<модуль>`, `triage`) и связь «relates to».
6. **Проверка.** Выполни каждую команду из «Как проверить», затем `make check` (и `node --check src/forte_guard/web/app.js` для UI). Пройди DoD из AGENTS.md 8.6 по пунктам. Метрики — только агрегаты; запрещённое из 2.2 не открывай.
7. **PR.** Коммиты `feat|fix|docs|test|chore: …` с `KAN-<N>` в теле, **без строк `Co-Authored-By` и других упоминаний агента как соавтора** (в коммитах и в описании PR); `git push -u origin <ветка>`; `gh pr create --title "KAN-<N>: …"` с разделами шаблона и чек-листом DoD; `gh pr checks --watch` до зелёного.
8. **Отчёт.** Комментарий по шаблону AGENTS.md 8.7 → переход в `In Review` (id 31). В `main` не мержить.
9. **Блокер или шаг человека** — по AGENTS.md 8.8, затем следующий тикет, если участник попросил продолжать.

В конце коротко скажи участнику: ключ тикета, ссылку на PR, что проверено, что осталось ему (ревью, мерж, шаги человека).
