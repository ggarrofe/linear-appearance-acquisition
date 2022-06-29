import json
import shutil
import glob

def copy_filtered_images(cameras_file_origin, prepare_dense_scene):
    with open(cameras_file_origin) as json_file:
        cameras = json.load(json_file)

    view_ids = [path.split("/")[-1].split(".exr")[0] for path in glob.glob(prepare_dense_scene+"*.exr")]
    print(view_ids)
    for view in cameras["views"]:
        
        if view["viewId"] in view_ids:
            path_parts = view["path"].split("/")
            path_parts[-2]=path_parts[-2]+"_filtered"
            print('/'.join(path_parts))
            shutil.copyfile(view["path"], '/'.join(path_parts))

if __name__ == "__main__":
    print("main")
    copy_filtered_images("./MeshroomCache/KnownCamera/535843973c2b1c1ce8d1e990ed5ea0d5c0f6fdbe/cameras.sfm", 
         "./MeshroomCache/PrepareDenseScene/eed26ebe42015c9b12d3bbe93d499a4587969369/")