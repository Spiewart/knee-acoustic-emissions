"""Setup configuration for acoustic_emissions_processing package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

# Read dev requirements
dev_requirements_path = Path(__file__).parent / "dev-requirements.txt"
dev_requirements = []
if dev_requirements_path.exists():
    dev_requirements = [
        line.strip()
        for line in dev_requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="acoustic_emissions_processing",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Tools for processing acoustic emissions data from biomechanical research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/acoustic_emissions_processing",
    packages=find_packages(include=["src", "src.*", "cli", "cli.*"]),
    package_dir={"": "."},
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "ae-sync-qc=cli.sync_qc:main",
            "ae-audio-qc=cli.audio_qc:main",
            "ae-visualize=cli.visualize:main",
            "ae-read-audio=cli.read_audio:main",
            "ae-process-directory=cli.process_directory:main",
            "ae-compute-spectrogram=cli.compute_spectrogram:main",
            "ae-add-inst-freq=cli.add_instantaneous_frequency:main",
            "ae-plot-per-channel=cli.plot_per_channel:main",
            "ae-dump-channels=cli.dump_channels_to_csv:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
)
