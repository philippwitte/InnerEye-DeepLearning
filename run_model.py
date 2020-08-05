#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import sys
from pathlib import Path

from azureml.core import Model, Run

import run_scoring

INNEREYE_SUBMODULE_NAME = "innereye-deeplearning"


def main() -> None:
    parser = argparse.ArgumentParser(description="Downloads and runs a model on images in a directory")
    parser.add_argument('--model-id', dest='model_id', action='store', type=str)
    known_args, unknown_args = parser.parse_known_args()
    workspace = Run.get_context().experiment.workspace
    model = Model(workspace=workspace, id=known_args.model_id)
    current_dir = Path(".")
    project_root = Path(model.download(str(current_dir))).absolute()
    # Remove --model-id and its value
    sys.argv = sys.argv[:1] + unknown_args
    run_scoring.run(project_root=project_root)


if __name__ == '__main__':
    main()
