#
# Train RFBNet on VOC2012 with Ignite
#
import os
from pathlib import Path


assert "VOC_ROOT" in os.environ or "VOC_DIR" in os.environ, \
    "Please define environment variable VOC_DIR or VOC_ROOT "
    "pointing VOCdevkit"

voc_root_var = "VOC_ROOT" if "VOC_ROOT" in os.environ else "VOC_DIR"
voc_root_path = Path(os.environ[voc_root_var])

assert voc_root_path.exists(), "Path '{}' is not found".format(voc_root_path)


def main():
    pass



if __name__ == "__main__":
    main()
