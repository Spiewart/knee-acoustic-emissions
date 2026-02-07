# PostgreSQL Setup (No Docker)

This guide documents native Postgres installs for local development on macOS and Windows.

## macOS

### Option A: Postgres.app (simplest)
1. Download and install: https://postgresapp.com/
2. Launch Postgres.app (it starts the server automatically).
3. Ensure `psql` is on your PATH (Postgres.app provides a menu for this).

### Option B: Homebrew
```bash
brew install postgresql@15
brew services start postgresql@15
```

Create a database:
```bash
createdb acoustic_emissions
```

## Windows

### Option A: Official Installer
1. Download from https://www.postgresql.org/download/windows/
2. Install with default options (include pgAdmin if desired).
3. Add the Postgres `bin` folder to your PATH so `psql` is available.

Create a database:
```powershell
createdb acoustic_emissions
```

## Connection String

Set a connection string in `.env.local`:
```
AE_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/acoustic_emissions
```

## Data Root Location

Set the default participant data root (optional):
```
AE_DATA_ROOT=/absolute/path/to/studies
```

Both `cli/process_directory.py` and `src/orchestration/participant.py` will use `AE_DATA_ROOT` if no path is provided on the CLI.
