# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'torchtt'
copyright = '2023, Ion Gabriel Ion'
author = 'Ion Gabriel Ion'
release = '2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon', 'sphinx.ext.intersphinx']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# import sphinx_bootstrap_theme

#html_theme = 'bootstrap'
#html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'https://github.com/ion-g-ion/torchTT/blob/main/logo_small.png?raw=true'

# #html_logo = "logo_small.png"
# 
# # Theme options are theme-specific and customize the look and feel of a
# # theme further.
# html_theme_options = {
#     # Navigation bar title. (Default: ``project`` value)
#     'navbar_title': "torchTT",
# 
#     # Tab name for entire site. (Default: "Site")
#     'navbar_site_name': "Site",
# 
#     # A list of tuples containing pages or urls to link to.
#     # Valid tuples should be in the following forms:
#     #    (name, page)                 # a link to a page
#     #    (name, "/aa/bb", 1)          # a link to an arbitrary relative url
#     #    (name, "http://example.com", True) # arbitrary absolute url
#     # Note the "1" or "True" value above as the third argument to indicate
#     # an arbitrary url.
#     'navbar_links': [
#         ("Installation guide", "./docs/install"),
#         ("Overview", "./docs/package-overview"),
#         ("Reference", "./docs/modules"),
#         ("Github", "https://github.com/ion-g-ion/torchTT", True),
#     ],
# 
#     # Render the next and previous page links in navbar. (Default: true)
#     'navbar_sidebarrel': False,
# 
#     # Render the current pages TOC in the navbar. (Default: true)
#     'navbar_pagenav': False,
# 
#     # Tab name for the current pages TOC. (Default: "Page")
#     # 'navbar_pagenav_name': "Page",
# 
#     # Global TOC depth for "site" navbar tab. (Default: 1)
#     # Switching to -1 shows all levels.
#     'globaltoc_depth': 2,
# 
#     # Include hidden TOCs in Site navbar?
#     #
#     # Note: If this is "false", you cannot have mixed ``:hidden:`` and
#     # non-hidden ``toctree`` directives in the same page, or else the build
#     # will break.
#     #
#     # Values: "true" (default) or "false"
#     'globaltoc_includehidden': "true",
# 
#     # HTML navbar class (Default: "navbar") to attach to <div> element.
#     # For black navbar, do "navbar navbar-inverse"
#     'navbar_class': "navbar navbar-inverse",
# 
#     # Fix navigation bar to top of page?
#     # Values: "true" (default) or "false"
#     'navbar_fixed_top': "true",
# 
#     # Location of link to source.
#     # Options are "nav" (default), "footer" or anything else to exclude.
#     # 'source_link_position': "nav",
# 
#     # Bootswatch (http://bootswatch.com/) theme.
#     #
#     # Options are nothing (default) or the name of a valid theme
#     # such as "cosmo" or "sandstone".
#     #
#     # The set of valid themes depend on the version of Bootstrap
#     # that's used (the next config option).
#     #
#     # Currently, the supported themes are:
#     # - Bootstrap 2: https://bootswatch.com/2
#     # - Bootstrap 3: https://bootswatch.com/3
#     'bootswatch_theme': "united",
# 
#     # Choose Bootstrap version.
#     # Values: "3" (default) or "2" (in quotes)
#     'bootstrap_version': "3",
# }