import pandas as pd
import numpy as np
import os
import glob
import CustomDataset

def int_to_4_bit_binary(value):
    # Ensure the value is within the allowed range
    if 0 <= value <= 15:
        return format(value, '04b')
    else:
        raise ValueError("Value is out of allowed range (0-4).")

def pad_string_to_x_bits(binary_str, binary_str_len = 32, padding_len= 12):
    # First, pad 12 zeros to the end
    binary_str = binary_str.ljust(len(binary_str) + padding_len, '0')

    # Then, pad zeros in front to make the total length 32
    binary_str = binary_str.rjust(binary_str_len, '0')

    return binary_str

def convert_directives_to_binary(file_path):
    # Reading the file
    with open(file_path, 'r') as file:
        file_content = file.readlines()

    # Displaying the first few lines of the file to get an idea of its content
    # file_content[:10]

    projects_data = {}
    current_project = None
    current_directive = None

    for line in file_content:
        line = line.strip()
        if line.startswith('open_project -reset'):
            # Extracting project name and initializing a new project list
            project_name = line.split('"')[1]
            current_project = project_name
            projects_data[current_project] = {}
        elif line.startswith(('set_directive_array_partition', 'set_directive_pipeline', 'set_directive_unroll')):
            # Starting a new directive row
            directive_parts = line.split()
            directive_name = directive_parts[0]
            
            if directive_name == 'set_directive_pipeline' or directive_name == 'set_directive_unroll':
                data_variable_name = directive_parts[-1].split('/')[1].strip('"')  # Data and project name is the last element, stripped of quotes
                data_project_name = directive_parts[-1].split('/')[0].strip('"') 
            else:
                data_variable_name = directive_parts[-1].strip('"')  # Data name is the last element, stripped of quotes
                data_project_name = directive_parts[-2].strip('"')  # Project name is the second last element, stripped of quotes
            
            current_directive = (directive_name, data_project_name, data_variable_name)
            
            if current_directive not in projects_data[current_project]:
                projects_data[current_project][current_directive] = []
            elif current_directive in projects_data[current_project]:
                current_directive = (directive_name, data_project_name, str(data_variable_name + '_new'))
                projects_data[current_project][current_directive] = []
                
            if directive_name == 'set_directive_pipeline':
                projects_data[current_project][current_directive].append(("pipeline", format(1, '04b')))
                
            command_parts = line.split(' ')
            flag = False
            
            for element in command_parts:
                if element.startswith('-') and element != '-core':
                    command = element
                    flag = True
                    continue
                if flag:
                    flag = False
                    value = element[:]
                    if command == '-type':
                        if value == 'block':
                            value = 0
                        elif value == 'cyclic':
                            value = 1
                        elif value == 'complete':
                            value = 2
                    value = int_to_4_bit_binary(int(value))
                    projects_data[current_project][current_directive].append((command, value))
        elif line.startswith('close_project'):
            # Closing the current project and resetting for the next project
            current_project = None
            current_directive = None

    # Displaying the projects data
    # for key, directives in projects_data.items():
    #     print(f"{key}:")
    #     for directive_key, commands in directives.items():
    #         commands_str = ', '.join([f"{command}: {value}" for command, value in commands])
    #         print(f"  {directive_key}: {commands_str}")
    #     print()
    
    # Step 1: Create an ordered list of commands for each directive
    directive_commands = {}
    for project in projects_data.values():
        for directive, commands in project.items():
            if directive not in directive_commands:
                directive_commands[directive] = []
            for command, _ in commands:
                if command not in directive_commands[directive]:
                    directive_commands[directive].append(command)

    # Step 2: Update each project with missing directives and commands
    for project_name, directives in projects_data.items():
        for directive, command_list in directive_commands.items():
            if directive not in directives:
                projects_data[project_name][directive] = [(cmd, int_to_4_bit_binary(0)) for cmd in command_list]
            else:
                existing_commands = set(cmd for cmd, _ in directives[directive])
                for cmd in command_list:
                    if cmd not in existing_commands:
                        directives[directive].append((cmd, int_to_4_bit_binary(0)))

    sorted_projects_data = {}
    for project_name, directives in projects_data.items():
        sorted_projects_data[project_name] = dict(sorted(directives.items()))
        
    # Displaying the updated projects data
    # for key, directives in sorted_projects_data.items():
    #     print(f"{key}:")
    #     for directive_key, commands in directives.items():
    #         commands_str = ', '.join([f"{cmd}: {value}" for cmd, value in commands])
    #         print(f"  {directive_key}: {commands_str}")
    #     print()
        
    # Update the projects data to binary data
    projects_binary_data = {}
    for project_name, directives in sorted_projects_data.items():
        directive_size = len(directives)
        if project_name not in projects_binary_data:
            projects_binary_data[project_name] = {}
        for directive, command_list in directives.items():
            commands_str = ''.join(f"{value}" for cmd, value in command_list)
            commands_str = pad_string_to_x_bits(commands_str)
            commands_str = [int(char) for char in commands_str]
            projects_binary_data[project_name][directive] = [commands_str]
                       
    # Displaying the updated projects data
    # for key, directives in projects_binary_data.items():
    #     print(f"{key}:")
    #     for directive_key, commands in directives.items():
    #         print(f"  {directive_key}: {commands}")
    #     print()
    
    return projects_binary_data, directive_size
    
def create_csv_files(Input, Input_with_directive, partname, benchmarkname, number_of_files_in_each_benchmark, number_of_variable_in_each_run, number_of_directive_in_each_run):
    try:
        os.makedirs("./Polybench_with_directive/" + benchmarkname +"/" + partname + "/", exist_ok=True)  # Make a directory to save created files.
    except OSError as error:
        print(error)
        
    i = 0
    for project_name, directives in Input_with_directive.items():
        data = []*((number_of_variable_in_each_run + number_of_directive_in_each_run)*32)
        for j in range(number_of_variable_in_each_run):
            value = str(Input[i][j+1])
            
            #check for input and output bits with "x" in the middle
            if "x" in value:
                index = value.find("x")
                value = str(int(value[0:index]) * int(value[index+1:]))
            data.extend(CustomDataset.float_bin(value))
            
        for directive, value in directives.items():
            data.extend(value[0])
        
        file_name = "./Inputs/Polybench_with_directive/" + benchmarkname +"/"+ partname + "/" + benchmarkname + str(eval("i")) + str(eval("Input[i][0]")) + "_" + str(number_of_directive_in_each_run) + ".csv" # This creates file_name
        np.savetxt(eval("file_name"), data, delimiter = ",", header = "V1", comments = "")
        i = i + 1
    print(str(eval("number_of_files_in_each_benchmark")) + " files"  + " created.")
    
if __name__ == "__main__":
    partname = "xc7v585tffg1157-3"
    benchmark_names = ["syrk", "syr2k","k3mm", "k2mm","gesummv", "gemm", "bicg"] # , "mvt", "atax"

    for benchmark_name in benchmark_names:
        Input = pd.read_csv("./Inputs/ML4Accel-Dataset-main/fpga_ml_dataset/HLS_dataset/polybench/" + benchmark_name +"/" + partname + "/" + "GAN_input_20.csv")
        file_path = "./Inputs/ML4Accel-Dataset-main/fpga_ml_dataset/HLS_dataset/polybench/" + benchmark_name +"/scripts/" + "hls.tcl"
        Input = Input.to_numpy()
        Input_with_directive, number_of_directive_in_each_run = convert_directives_to_binary(file_path)

        number_of_variable_in_each_run = Input.shape[1]-1
        number_of_files_in_each_benchmark = Input.shape[0]

        create_csv_files(Input, Input_with_directive, partname, benchmark_name, number_of_files_in_each_benchmark, number_of_variable_in_each_run, number_of_directive_in_each_run)