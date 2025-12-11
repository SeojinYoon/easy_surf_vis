
import os

def exec_command_inConda(command, 
                         conda_env_name,
                         conda_env_path = "/home/name/anaconda3/condabin",
                         arg_symbol = "-",
                         parameter_info = {}, 
                         pipeline_info = {},
                         is_print = True,
                         capture_output = False):
    """
    Execute command in conda environment
    
    :param command: command
    :param parameter_info(dictionary): argument information
        -k [argument_name]: [argument_value]
        -k 1, [value]: if there is not arg name, set the name using number
    :param pipeline_info(dictionary): pipeline
    :param is_print(boolean): flag for printing command
    :param capture_output(boolean): flag for getting output
    
    return: result of command
    """ 
    os.environ["PATH"] += f":{conda_env_path}"
    
    active_conda_command = f"conda run -n {conda_env_name} --no-capture-output "
    origin_command = make_command(command = command, 
                                  arg_symbol = arg_symbol, 
                                  parameter_info = parameter_info, 
                                  pipeline_info = pipeline_info)
    
    commands = " ".join([active_conda_command, origin_command])
    if is_print:
        print(f"\033[1m" + commands + "\033[0m")
    
    result = os.system(commands)
    return result

def make_command(command, 
                 arg_symbol = "-",
                 parameter_info = {}, 
                 pipeline_info = {}):
    """
    Make command
    
    :param command: command
    :param parameter_info(dictionary): argument information
        -k [argument_name]: [argument_value]
        -k 1, [value]: if there is not arg name, then set the name using number
    :param pipeline_info(dictionary): pipeline
    """
    parameter_info = {} if parameter_info is None else parameter_info
    pipeline_info = {} if pipeline_info is None else pipeline_info
    
    arg_str = ""
    for argument in parameter_info:
        arg_type = type(argument)
        
        value = parameter_info[argument]
        value_type = type(value)
        
        # Value parsing
        value_str = ""
        if value_type == list:
            if len(value) > 0:
                for e in value:
                    value_str += str(e) + " "
                
        elif value_type == str:
            if value_type != "":
                value_str = value                
        else:
            value_str = str(value)
        
        # Make arg str
        if arg_type == str:
            if value_str == "":
                arg_str += f" {arg_symbol}" + argument
            else:
                arg_str += f" {arg_symbol}" + argument + " " + value_str
        elif arg_type == tuple:
            for a, v in zip(argument, value):
                arg_str += f" {arg_symbol}" + a + " " + v
        else:
            arg_str += " " + value_str
        
    command = command + arg_str
    
    for pipe in pipeline_info:
        command += " " + pipe + " " + pipeline_info[pipe]
    return command
    