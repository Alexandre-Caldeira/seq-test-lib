% SCRIPT: combine_m_files.m
% PURPOSE: Reads all .m files in a user-selected folder and concatenates
%          their content into a single .txt file, separated by headers.

clear; clc; close all;

% --- Configuration ---
outputFileName = 'combined_m_files.txt'; % Name of the output text file

% --- 1. Select the Folder ---
folderPath = uigetdir(pwd, 'Select the folder containing .m files');

% Check if the user cancelled the dialog
if isequal(folderPath, 0)
    disp('Folder selection cancelled by user. Exiting script.');
    return; % Exit the script
end

fprintf('Selected folder: %s\n', folderPath);

% --- 2. Find all .m Files in the Selected Folder ---
filePattern = fullfile(folderPath, '*.m'); % Create a pattern to search for .m files
mFiles = dir(filePattern);                 % Get directory listing matching the pattern

% Check if any .m files were found
if isempty(mFiles)
    fprintf('No .m files found in the selected folder: %s\n', folderPath);
    return; % Exit if no files to process
end

fprintf('Found %d .m file(s) to process.\n', length(mFiles));

% --- 3. Prepare the Output File ---
outputFilePath = fullfile(folderPath, outputFileName); % Full path for the output file

% Open the output file for writing in text mode ('wt')
% 'wt' ensures correct line endings across different operating systems.
outputFileID = fopen(outputFilePath, 'wt');

% Check if the output file could be opened
if outputFileID == -1
    error('Cannot open output file "%s" for writing. Check permissions.', outputFilePath);
    % Error will stop the script execution
end

% Use onCleanup to ensure the output file is closed even if errors occur
cleanupOutput = onCleanup(@() fclose(outputFileID));

fprintf('Writing combined content to: %s\n', outputFilePath);

% --- 4. Process Each .m File ---
totalFilesProcessed = 0;
for i = 1:length(mFiles)
    currentMFileName = mFiles(i).name;
    currentMFilePath = fullfile(folderPath, currentMFileName);

    fprintf('Processing: %s...\n', currentMFileName);

    try
        % --- Write Header for the Current File ---
        fprintf(outputFileID, '============================================================\n');
        fprintf(outputFileID, '=== FILE START: %s\n', currentMFileName);
        fprintf(outputFileID, '============================================================\n\n');

        % --- Open the current .m file for reading ('rt') ---
        inputFileID = fopen(currentMFilePath, 'rt');
        if inputFileID == -1
            warning('Could not open file: %s. Skipping this file.', currentMFilePath);
            fprintf(outputFileID, '*** ERROR: Could not read file %s ***\n\n', currentMFileName);
            continue; % Skip to the next file in the loop
        end
        % Ensure the input file is closed after reading or if an error occurs within the loop
        cleanupInput = onCleanup(@() fclose(inputFileID));

        % --- Read and Write Content Line by Line ---
        while ~feof(inputFileID)
            line = fgetl(inputFileID); % Read one line (without newline chars)
            if ischar(line) % fgetl returns -1 at EOF, which is not a char
                fprintf(outputFileID, '%s\n', line); % Write the line + a newline
            end
        end

        % --- Add Space After File Content ---
        fprintf(outputFileID, '\n\n'); % Add two newlines for separation

        % Explicitly delete the input file cleanup object for this iteration
        % (fclose(inputFileID) will be called by it)
        clear cleanupInput;
        totalFilesProcessed = totalFilesProcessed + 1;

    catch ME % Catch potential errors during file processing
        warning('An error occurred while processing file %s: %s. Skipping rest of this file.', ...
                currentMFileName, ME.message);
        fprintf(outputFileID, '\n*** ERROR occurred during processing of file %s ***\n\n', currentMFileName);
        % Ensure input file handle is closed if error happened after opening
        if exist('inputFileID', 'var') && inputFileID ~= -1 && strcmp(fopen(inputFileID), currentMFilePath)
             fclose(inputFileID);
        end
         clear cleanupInput; % Make sure onCleanup doesn't try to double-close
    end
end

% --- 5. Finalization ---
% Output file is closed automatically by the 'cleanupOutput' onCleanup object
% when the script finishes or errors out.

fprintf('------------------------------------------------------------\n');
if totalFilesProcessed == length(mFiles)
    fprintf('Successfully processed %d .m files.\n', totalFilesProcessed);
else
     fprintf('Processed %d out of %d .m files. Some files may have been skipped due to errors (see warnings).\n', ...
             totalFilesProcessed, length(mFiles));
end
fprintf('Combined output written to: %s\n', outputFilePath);
fprintf('------------------------------------------------------------\n');

% Optional: Open the generated text file
% uncomment the next line if you want the file to open automatically
% edit(outputFilePath);