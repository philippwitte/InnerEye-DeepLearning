#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import sys
from pathlib import Path

from azureml.core import Model, Run


INNEREYE_SUBMODULE_NAME = "innereye-deeplearning"


def main() -> None:
    parser = argparse.ArgumentParser(description="Downloads and runs a model on images in a directory")
    parser.add_argument('--model-id', dest='model_id', action='store', type=str)
    known_args, unknown_args = parser.parse_known_args()
    workspace = Run.get_context().experiment.workspace
    model = Model(workspace=workspace, id=known_args.model_id)
    project_root = Path(__file__).parent.absolute()
    model_path = Path(model.download(str(project_root)))
    # Move everything in model_path to project_root, where possible
    for src in sorted(model_path.glob("*")):
        dst = project_root / src.relative_to(model_path)
        if not dst.exists():
            src.rename(dst)
        elif dst != Path(__file__):
            print(f"WARNING: not moving {src} to {dst} as it already exists")
    runner = project_root / INNEREYE_SUBMODULE_NAME / "run_scoring.py"
    if not runner.exists():
        print(f"WARNING: not found: {runner}")
    for path in [project_root, project_root / INNEREYE_SUBMODULE_NAME]:
        name = str(path)
        if path.exists() and name not in sys.path:
            sys.path.append(name)
    import run_scoring
    sys.argv = sys.argv[:1] + unknown_args
    run_scoring.run(project_root=project_root)


if __name__ == '__main__':
    main()
