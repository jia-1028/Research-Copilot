# GitHub 发布前检查清单

本仓库设计为可公开的代码副本。运行数据和私有研究素材应始终保留在本机，不进入 Git 历史。

## 已在此副本完成

- `.env`、数据库、Chroma、原始 PDF、解析产物、上传文件、日志和报告均由 `.gitignore` 排除；
- `.env.example` 只含空配置位，不含真实密钥；
- 本地绝对路径、特定论文的批量导入逻辑与私有评测集已移除；
- README、演示和测试文档使用通用论文示例；
- GitHub Actions 仅运行离线 lint 与测试，不使用任何 API Key；
- 本副本将使用全新的 Git 历史，避免旧提交中可能保留的私有素材。

## 发布者仍需确认

1. 在 GitHub 创建空仓库，不要勾选自动创建 README、`.gitignore` 或 License；
2. 选择适用的开源许可证并在仓库根目录添加 `LICENSE`；许可证决定他人能否使用、修改和分发代码；
3. 再次检查 `git status --ignored`，确认 `.env`、`data/` 运行文件和 PDF 没有进入暂存区；
4. 检查代码、文档、截图和 Git 历史中不存在 API Key、令牌、私人路径、未公开论文内容或个人信息；
5. 在仓库 Settings 中启用 secret scanning、push protection 与 Dependabot alerts（可用时）；
6. 首次推送后，在全新目录执行 clone、安装、离线测试，确认 README 可复现。

## 首次推送命令

将 `<YOUR_REPOSITORY_URL>` 换成你新建的空 GitHub 仓库 HTTPS 或 SSH 地址：

```powershell
Set-Location <YOUR_LOCAL_REPOSITORY_PATH>
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

推送前可以用以下命令审查暂存内容：

```powershell
git status
git diff --cached --check
git ls-files | Select-String -Pattern '(^|/)(\.env|.*\.pdf|.*\.db|.*\.sqlite)$'
```

若上述最后一条命令有输出，请停止推送并先移除对应文件。即使文件后来被删除，只要它已经进入公开 Git 历史，也需要视为可能泄露并按情况进行历史清理和密钥轮换。
