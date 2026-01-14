"""Quality Assurance Checks"""

import os
import sys
from pathlib import Path

def check_package_structure():
    """Check package structure integrity"""
    required_files = [
        "src/pricechecktn/__init__.py",
        "src/pricechecktn/data/__init__.py",
        "src/pricechecktn/features/__init__.py",
        "src/pricechecktn/models/__init__.py",
        "src/pricechecktn/pipelines/__init__.py",
        "src/pricechecktn/scraping/__init__.py",
        "src/pricechecktn/utils/__init__.py",
        "src/pricechecktn/api/__init__.py",
        "mlops/experiment_tracking/__init__.py",
        "mlops/model_registry/__init__.py",
        "mlops/orchestration/__init__.py"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f" Missing files: {missing_files}")
        return False

    print(" Package structure OK")
    return True

def check_requirements():
    """Check requirements files"""
    required_requirements = ["requirements.txt", "requirements-api.txt"]

    missing_requirements = []
    for req_file in required_requirements:
        if not Path(req_file).exists():
            missing_requirements.append(req_file)

    if missing_requirements:
        print(f" Missing requirements files: {missing_requirements}")
        return False

    print(" Requirements files OK")
    return True

def check_documentation():
    """Check documentation files"""
    required_docs = [
        "mlops/experiment_tracking/README.md",
        "mlops/model_registry/README.md",
        "mlops/orchestration/README.md"
    ]

    missing_docs = []
    for doc_file in required_docs:
        if not Path(doc_file).exists():
            missing_docs.append(doc_file)

    if missing_docs:
        print(f" Missing documentation: {missing_docs}")
        return False

    print(" Documentation OK")
    return True

def run_all_checks():
    """Run all QA checks"""
    print(" Running Quality Assurance Checks...")

    checks = [
        check_package_structure(),
        check_requirements(),
        check_documentation()
    ]

    if all(checks):
        print(" All QA checks passed!")
        return True
    else:
        print(" Some QA checks failed!")
        return False

if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)