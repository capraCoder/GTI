@echo off
REM ============================================
REM GTI v3.0 - Git Commit & Push Script
REM ============================================

cd /d D:\CaprazliIndex\GTI

echo.
echo ========================================
echo   GTI v3.0 - Committing to GitHub
echo ========================================
echo.

REM Check if git is initialized
if not exist ".git" (
    echo [!] Git not initialized. Initializing...
    git init
    git remote add origin https://github.com/capraCoder/GTI.git
    echo [+] Remote set to: https://github.com/capraCoder/GTI.git
    echo.
)

echo [*] Staging files...
git add app.py
git add gti_engine.py
git add canonical_cases.yaml 2>nul
git add requirements.txt 2>nul
git add README.md 2>nul
git add .gitignore 2>nul

echo.
echo [*] Files staged:
git status --short

echo.
echo [*] Creating commit...
git commit -m "feat(v3.0): Out_of_Scope handling + View Layer toggle + Orthogonal types

- Added Out_of_Scope panel for non-2x2 games (zero-sum, sequential, n-player)
- Implemented View Layer toggle (Public Narrative vs Revealed Reality)
- Enforced 12 canonical Robinson-Goforth orthogonal game types
- Fixed Pydantic validation for edge cases
- Added example buttons in sidebar (including Matching Pennies test)
- Improved deception detection UI with evidence categories"

echo.
echo [*] Pushing to GitHub (private repo)...
git branch -M main
git push -u origin main

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Push failed. You may need to authenticate.
    echo [!] Try: git push -u origin main
    echo.
)

echo.
echo ========================================
echo   Done! Check: https://github.com/capraCoder/GTI
echo ========================================
echo.
pause
