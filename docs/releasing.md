# Releasing to PyPI

Releases are built and uploaded by the `Release` workflow
(`.github/workflows/release.yml`), which triggers on any pushed tag matching
`v*`. It builds an sdist and a pure-Python wheel, validates the metadata with
`twine check`, and uploads to PyPI using
[trusted publishing](https://docs.pypi.org/trusted-publishers/) — there is no
API token stored anywhere in the repository.

## One-time setup

These steps only need to be done once, by someone with owner rights on the
`simonsobs/soma` GitHub repository and on the PyPI project.

### 1. Create the GitHub environment

In the GitHub repository, go to **Settings → Environments → New environment**
and create one named `pypi`. The `publish` job references this environment, so
the name must match exactly.

Optionally add required reviewers to the environment. Doing so pauses every
release until an approver signs off, which is a useful safety net.

### 2. Register the trusted publisher on PyPI

Trusted publishing lets PyPI accept uploads from a specific GitHub workflow via
short-lived OpenID Connect (OIDC) tokens, so no long-lived secret is needed.

If the project does **not** exist on PyPI yet, add a *pending* publisher at
<https://pypi.org/manage/account/publishing/>. If it already exists, go to
**Your projects → soma → Manage → Publishing → Add a new publisher**.

Fill in the GitHub form with:

| Field                | Value           |
| -------------------- | --------------- |
| Owner                | `simonsobs`     |
| Repository name      | `soma`          |
| Workflow name        | `release.yml`   |
| Environment name     | `pypi`          |

The values must match the repository and workflow exactly, otherwise PyPI
rejects the upload with a `403 invalid-publisher` error.

### 3. (Recommended) Do the same on TestPyPI

Repeat step 2 at <https://test.pypi.org/manage/account/publishing/> if you want
to rehearse a release. To publish there, add `repository-url:
https://test.pypi.org/legacy/` under `with:` in the
`pypa/gh-action-pypi-publish` step.

### Why not an API token?

An API token (`PYPI_API_TOKEN` as a repository secret, passed to the publish
action as `password`) also works and is the fallback if trusted publishing
cannot be used. It is less preferable because the token is long-lived, has to
be rotated manually, and is readable by any workflow with access to secrets.

## Cutting a release

The version is stored in exactly one place: the `version` field of
`pyproject.toml`. `soma.__version__` reads it back at runtime through
`importlib.metadata`, so it never needs to be edited separately.

The project follows [semantic versioning](https://semver.org): bump the patch
number for bug fixes, the minor number for backwards-compatible features, and
the major number for breaking changes.

1. Make sure `main` is green in CI and that you are up to date:

   ```bash
   git switch main
   git pull
   ```

2. Bump the version in `pyproject.toml`, for example from `0.0.1` to `0.1.0`.

3. Commit and push the bump on a pull request, and merge it once CI passes:

   ```bash
   git commit -am "Bump version to 0.1.0"
   ```

4. Tag the merge commit. The tag must be the version prefixed with `v`:

   ```bash
   git switch main && git pull
   git tag -a v0.1.0 -m "soma 0.1.0"
   git push origin v0.1.0
   ```

5. Watch the `Release` workflow in the Actions tab. If the `pypi` environment
   has required reviewers, approve the run when prompted.

6. Verify the upload:

   ```bash
   pip install --upgrade soma
   python -c "import soma; print(soma.__version__)"
   ```

7. Optionally publish a GitHub Release for the tag with the changelog.

## Notes and troubleshooting

- **The tag and `pyproject.toml` must agree.** Nothing enforces this
  automatically; a mismatch will publish a version that differs from the tag.
- **PyPI versions are immutable.** A published version can be yanked but never
  replaced. If a release is broken, bump the patch version and release again.
- **`Version already exists` (400).** The version in `pyproject.toml` was not
  bumped since the last release.
- **`invalid-publisher` (403).** The owner, repository, workflow filename or
  environment on PyPI does not match the workflow that is running.
- **Read the Docs versions.** Enable the *Build pull requests* and tag-based
  versions in the Read the Docs project settings so that each tag gets its own
  documentation version.
