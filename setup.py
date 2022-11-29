from setuptools import setup

setup(
    use_scm_version={
        "version_scheme": "post-release",
        "write_to": "eqgat/_version.py",
    },
    setup_requires=["setuptools_scm"],
)
