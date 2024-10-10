import os
import argparse
import xarray as xr
import pandas as pd
from multiprocessing import Process

def process_folder(folder):
    base_src_dir = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/yzy/downscaling_precipitation/precip_dataset"
    base_dst_dir = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/yzy/downscaling_precipitation/precip_dataset_new"

    src_dir = os.path.join(base_src_dir, folder)
    dst_dir = os.path.join(base_dst_dir, folder)
    os.makedirs(dst_dir, exist_ok=True)  # 确保目标目录存在

    # 遍历目录下的所有.nc文件
    for filename in os.listdir(src_dir):
        if filename.endswith('.nc'):
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, filename)
            
            # 读取数据
            ds = xr.open_dataset(src_file)
            
            # 重新获取第一个和最后一个时间点
            first_time_point = ds.time.isel(time=0).values
            last_time_point = ds.time.isel(time=-1).values

            # 删除第一个小时的数据并平移时间
            ds_shifted = ds.sel(time=slice(first_time_point, None)).drop_sel(time=first_time_point)
            ds_shifted = ds_shifted.assign_coords(time=ds_shifted.time - pd.Timedelta(hours=1))

            # 删除最后一个小时的数据
            ds_trimmed = ds.sel(time=slice(None, last_time_point)).drop_sel(time=last_time_point)

            # 合并处理后的数据集
            vars_to_shift = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in", "u700_in", "v700_in"]
            vars_to_keep = ["yw_hourly_in", "yw_hourly_tar"]
            ds_final = xr.merge([ds_shifted[vars_to_shift], ds_trimmed[vars_to_keep]])

            # 保存修正后的文件
            ds_final.to_netcdf(dst_file)

            print(f"Processed and saved: {dst_file}")

def main():
    parser = argparse.ArgumentParser(description="Process .nc files in specified folder.")
    parser.add_argument("folder", type=str, help="Folder to process: test, test_small, train, or train_small")
    args = parser.parse_args()

    process_folder(args.folder)

if __name__ == "__main__":
    main()
