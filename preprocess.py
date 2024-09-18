import os
import re

def remove_comments_and_imports(java_code):
    """
    Removes comments and import statements from Java code.
    """
    # Remove single-line comments (//)
    code_no_single_line_comments = re.sub(r'//.*', '', java_code)
    
    # Remove multi-line comments (/* ... */)
    code_no_comments = re.sub(r'/\*.*?\*/', '', code_no_single_line_comments, flags=re.DOTALL)
    
    # Remove import statements
    code_no_imports = re.sub(r'import\s+.*?;\s*', '', code_no_comments)
    
    return code_no_imports

def process_java_files(directory, output_file):
    """
    Process all Java files in the given directory and save the cleaned content to a single text file.
    """
    # List all Java files in the directorypath/to/your/java/files
    java_files = [f for f in os.listdir(directory) if f.endswith('.java')]
    
    all_cleaned_code = ''
    
    for java_file in java_files:
        file_path = os.path.join(directory, java_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            java_code = file.read()
            cleaned_code = remove_comments_and_imports(java_code)
            all_cleaned_code += cleaned_code + '\n\n'  # Add a newline between files for separation
    
    # Save the cleaned code to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(all_cleaned_code)
    
    print(f"Processed {len(java_files)} Java files. Cleaned content saved to {output_file}")

if __name__ == '__main__':
    directory = '/home/akarsh/Desktop/DATASET/TRAINING'  # Update this path to your Java files directory
    output_file = 'cleaned_java_dataset.txt'  # Output file to save the cleaned content
    
    process_java_files(directory, output_file)
