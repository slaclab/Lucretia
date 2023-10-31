function [ stripedName ] = convertSDDSname( name )
% ************************************************************************
% Copyright (c) 2002 The University of Chicago, as Operator of Argonne
% National Laboratory.
% Copyright (c) 2002 The Regents of the University of California, as
% Operator of Los Alamos National Laboratory.
% This file is distributed subject to a Software License Agreement found
% in the file LICENSE that is included with this distribution. 
% ************************************************************************
% Converts SDDS parameter and column names to valid MATLAB variable names
    stripedName = strrep(name, ':', '_COLON_');
    stripedName = strrep(stripedName, '~', '_TILDE_');
    stripedName = strrep(stripedName, '/', '_FSLASH_');
    stripedName = strrep(stripedName, '\', '_BSLASH_');
    stripedName = strrep(stripedName, '`', '_FQUOTE_');
    stripedName = strrep(stripedName, '!', '_EXCLAMATION_');
    stripedName = strrep(stripedName, '.', '_PERIOD_');
    stripedName = strrep(stripedName, '@', '_AT_');
    stripedName = strrep(stripedName, '#', '_POUND_');
    stripedName = strrep(stripedName, '$', '_DOLLAR_');
    stripedName = strrep(stripedName, '%', '_PERCENT_');
    stripedName = strrep(stripedName, '^', '_HAT_');
    stripedName = strrep(stripedName, '&', '_AMPERSAND_');
    stripedName = strrep(stripedName, '*', '_STAR_');
    stripedName = strrep(stripedName, '(', '_LBRACE_');
    stripedName = strrep(stripedName, ')', '_RBRACE_');
    stripedName = strrep(stripedName, '-', '_DASH_');
    stripedName = strrep(stripedName, '=', '_EQUALS_');
    stripedName = strrep(stripedName, '+', '_PLUS_');
    stripedName = strrep(stripedName, '[', '_SQLBRACE_');
    stripedName = strrep(stripedName, ']', '_SQRBRACE_');
    stripedName = strrep(stripedName, '{', '_LSQUIGGLE_');
    stripedName = strrep(stripedName, '}', '_RSQUIGGLE_');
    stripedName = strrep(stripedName, '|', '_VERTLINE_');
    stripedName = strrep(stripedName, ';', '_SEMICOLON_');
    stripedName = strrep(stripedName, '"', '_DOUBLEQUOTE_');
    stripedName = strrep(stripedName, '<', '_LESSTHAN_');
    stripedName = strrep(stripedName, '>', '_GREATERTHAN_');
    stripedName = strrep(stripedName, ',', '_COMMA_');
    stripedName = strrep(stripedName, '?', '_QUESTION_');


