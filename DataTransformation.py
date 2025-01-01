import pandas as pd
import numpy as np
import os
import glob

# Transform float and int number into a 32-bit fixed point 
# binary number with 12 bits as the fraction part
def float_bin(number, places = 12):
    resstr = ""
    if number.find('.') != -1 :
        # split() separates whole number and decimal
        whole, dec = str(number).split(".")

        # Convert both whole number and decimal
        whole = int(whole)
        dec = int (dec)
        dec = decimal_converter(dec)
        
        res = [0]*32
        i = 31 - places
        
        while whole > 0 and i >= 0 :
            res[i] = int (whole % 2)
            whole = int (whole / 2)
            i -= 1
        
        # Convert the whole number part to it's
        # respective binary form
        
        i = 31
        dec = dec * 2**12
        while dec > 0 and i >= 20 :
            res[i] = int (dec % 2)
            dec = int (dec / 2)
            i -= 1
        resstr = ''.join(str(x) for x in res)
        
    else :
        whole = int(number)
        
        res = [0]*32
        i = 31 - places
        
        while whole > 0 and i >= 0 :
            res[i] = int (whole % 2)
            whole = int (whole / 2)
            resstr += str (res[i])
            i -= 1
            
        resstr = ''.join(str(x) for x in res)
    return res
 
# Function converts the value passed as
# parameter to it's decimal representation
def decimal_converter(num):
    while num > 1:
        num /= 10
    return num

# Create file for each design point and save the transformed number
def create_csv_files(Input, partname, benchmarkname, number_of_files_in_each_benchmark, number_of_variable_in_each_run):
    try:
        os.makedirs("./Inputs/20_polybench/" + benchmarkname +"/" + partname + "/", exist_ok=True)  # Make a directory to save created files.
    except OSError as error:
        print(error)
    for i in range(number_of_files_in_each_benchmark):
        data = []*(number_of_variable_in_each_run*32)
        for j in range(number_of_variable_in_each_run):
            value = str(Input[i][j+1])
            
            #check for input and output bits with "x" in the middle
            if "x" in value:
                index = value.find("x")
                value = str(int(value[0:index]) * int(value[index+1:]))
            data.extend(float_bin(value))
        file_name = "./Inputs/20_polybench/" + benchmarkname +"/"+ partname + "/" + benchmarkname + str(eval("i")) + str(eval("Input[i][0]")) + "_20.csv" # This creates file_name
        np.savetxt(eval("file_name"), data, delimiter = ",", header = "V1", comments = "")
    print(str(eval("number_of_files_in_each_benchmark")) + " files"  + " created.")

# Transform data to binary representation
partname = "xc7v585tffg1157-3"
benchmark_names = ["syrk", "syr2k", "mvt", "k3mm", "k2mm","gesummv", "gemm", "bicg", "atax"]

for benchmark_name in benchmark_names:
    Input = pd.read_csv("./Inputs/ML4Accel-Dataset-main/fpga_ml_dataset/HLS_dataset/polybench/" + benchmark_name +"/" + partname + "/" + "GAN_input_20.csv")
    Input = Input.to_numpy()

    number_of_variable_in_each_run = Input.shape[1]-1
    number_of_files_in_each_benchmark = Input.shape[0]
    
    create_csv_files(Input,partname, benchmark_name, number_of_files_in_each_benchmark, number_of_variable_in_each_run)
    
i = 0
for benchmark_name in benchmark_names:
    print(benchmark_name)
    if i == 0 :
        files = glob.glob("./Inputs/20_polybench/" + benchmark_name + "/" + partname + "/*")
        i += 1
    else: 
        files += glob.glob("./Inputs/20_polybench/" + benchmark_name + "/" + partname + "/*")
print("Total number of files: ", len(files))
print("Showing first 10 files...")
files[:10]