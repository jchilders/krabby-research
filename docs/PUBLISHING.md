# Publishing to PyPI

Short checklist for publishing **krabby‑*** packages to PyPI using the existing GitHub Actions workflow.

TODO: Complete credential setup (PyPI token or Trusted Publishing) to enable uploads.

**Workflow file:** `.github/workflows/publish-packages.yml`

**The workflow has been dry‑run** by pushing a tag and verifying all steps pass except the final PyPI upload (which fails until credentials are added) :  [Example run](https://github.com/flliver/krabby-research/actions/runs/21758235909)
  ```
  git tag controller-v0.1.0 
  git push origin controller-v0.1.0
  ``` 
 

## Tag patterns and package names

| Tag pattern | PyPI package |
| :-- | :-- |
| `hal-client-v*` | krabby-hal-client |
| `hal-server-v*` | krabby-hal-server |
| `compute-parkour-v*` | krabby-compute-parkour |
| `controller-v*` | krabby-controller |
| `hal-tools-v*` | krabby-hal-tools |
| `hal-server-isaac-v*` | krabby-hal-server-isaac |
| `hal-server-jetson-v*` | krabby-hal-server-jetson |

> **Note:** More specific patterns override generic ones. For example, `hal-server-isaac-v0.1.0` → `krabby-hal-server-isaac`, not `krabby-hal-server`.

## Account

- Create an account at [pypi.org](https://pypi.org).

## Token

- In PyPI: Account → API tokens.
- Create a token; scope it to the project(s) you publish or to the whole account.
- Keep the token secret; do not commit it.

## GitHub secrets

- Repo → Settings → Secrets and variables → Actions.
- Add a secret: `PYPI_API_TOKEN` with the PyPI API token value.
- The workflow uses `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=${{ secrets.PYPI_API_TOKEN }}`.

## Trusted Publishing (optional)

- Link the GitHub repo to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
- After one-time setup, the workflow can upload without a long-lived token.

## Publishing

- Push a version tag to trigger the workflow, e.g.:
  - `git tag hal-client-v0.1.0 && git push origin hal-client-v0.1.0`
- CI builds the package, runs its tests, then uploads to PyPI.
- **Publish in dependency order** so dependents can install from PyPI:
  1. `hal-client-v*`, `hal-server-v*` (no internal deps)
  2. `compute-parkour-v*`, `controller-v*`, `hal-tools-v*`
  3. `hal-server-isaac-v*`, `hal-server-jetson-v*`


## First-time: reserve package names

- Reserve each name on PyPI so no one else can use it.
- Either create a minimal release (e.g. push a tag and let CI publish) or use the PyPI web UI to create the project.
- Package names to reserve: `krabby-hal-client`, `krabby-hal-server`, `krabby-compute-parkour`, `krabby-controller`, `krabby-hal-tools`, `krabby-hal-server-isaac`, `krabby-hal-server-jetson`.

---

## Appendix

### (a) How this works

The workflow runs **only when you push a tag** (not on every branch push). After you push a tag that matches a pattern (e.g. `controller-v0.1.0`):

1. GitHub Actions checks out the repo and parses the tag to pick the package (path, dependency list, test path).
2. It builds and installs any internal krabby-* dependency wheels from the repo (in order).
3. It builds the package wheel, installs it and pytest, then runs that package’s tests.
4. If tests pass, it uploads the wheel to PyPI with `twine` using `PYPI_API_TOKEN`.

More specific tag patterns (e.g. `hal-server-isaac-v*`) are matched before generic ones (e.g. `hal-server-v*`) so the right package is chosen.

### (b) How to create the tag

Create the tag locally, then push it:

```bash
git tag <tag-name>              # e.g. git tag controller-v0.1.0
git push origin <tag-name>     # e.g. git push origin controller-v0.1.0
```

To tag a specific commit: `git tag <tag-name> <commit-hash>`. List tags: `git tag` or `git tag -l 'controller-v*'`.

### (c) How to test locally without PyPI

From the repo root, run the same steps the workflow runs for one package (e.g. controller):

```bash
python3.12 -m venv testenv && source testenv/bin/activate

pip install --upgrade pip && pip install build twine pytest pytest-cov

cd hal/client && python -m build --wheel && pip install dist/*.whl && cd ../..

cd controller && python -m build --wheel && pip install dist/*.whl && cd ..

pytest tests/unit/controller/ -v
```

If this passes, the workflow’s build and test steps will work. No tag or PyPI needed.
