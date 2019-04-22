% 
function [filepaths, filenames, cleanFilenames] = getFilepaths(dirPattern)
% GETFILEPATHS Returns fullfilepaths and filenames (no extentions)
% Inputs:
%   dirPattern : pattern to use for dir call
% Outputs:
%   filepaths: cell array of full filepaths
%   filenames: cell array fo filenames, no extension
%
    filepaths = dir(dirPattern);
    filenames = {filepaths.name}';
    % filenames no extension
    filenames = regexprep(filenames,'\..*$','');
    filepaths = strcat({filepaths.folder}',filesep,{filepaths.name}');
    cleanFilenames = regexprep(filenames,'[^a-zA-Z0-9-]','');
    cleanFilenames = regexprep(cleanFilenames,'-','_');
end