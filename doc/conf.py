# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from importlib.metadata import version

project = "Active Space Finder"
copyright = "Copyright © 2020-2024 HQS Quantum Simulations GmbH. All Rights Reserved."
author = "HQS Quantum Simulations GmbH"
release = version("active-space-finder")
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.coverage", "sphinx.ext.napoleon", "nbsphinx"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Extension configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/index.html

# autodoc
autodoc_member_order = "alphabetical"
"""Specifies how automatically documented members are sorted: 'alphabetical' (default),
'groupwise', bysource'."""

autodoc_mock_imports = ["pyscf"]
"""A list of modules to prevent import errors from halting the building process when some external
dependencies are not importable at build time."""

# nbsphinx
nbsphinx_execute = "always"
"""Explicitly dis-/enabling notebook execution. Possible options: 'always', 'auto', 'never'."""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_css_files = ["theme_mod.css"]
html_logo = "media/HQS.jpg"
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
html_theme_options = {"style_nav_header_background": "white"}
