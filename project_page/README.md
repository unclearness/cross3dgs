# Project page

Static GitHub Pages site for Cross3DGS, built on the
[Nerfies](https://github.com/nerfies/nerfies.github.io) template (CC BY-SA 4.0).
`.github/workflows/pages.yml` deploys this directory from the `main`
branch of the public repository; set Settings > Pages > Source to "GitHub Actions" once.

Preview locally: `python -m http.server -d project_page 8000`.

## Release media

Capture examples use the teddy bear. The outdoor video shows a shrine and a
stone statue; pedestrians and vehicles in the stone-statue recording are
blurred in post-processing. Before publication, run
`python scripts/verify_release_media.py` from the repository root. The Pages
workflow checks the exact media inventory and hashes in [media_manifest.json](media_manifest.json).
Review replacement images, posters, and the full video before updating this manifest.
