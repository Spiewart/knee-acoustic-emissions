function read_audio_board_file(fname, outputFolder)
%% READ_AUDIO_BOARD_FILE 
% Reads the binary file output from the audio board. It intreprets the
% binary and outputs a *.mat file (with the same name as fname) that
% contains the data from the microphones and header.
%
% Inputs:
%   - fname: [string] absolute or relative file path of the binary file
%   from the device. This must be the *.bin file from the HeartPulse app.
%   - outputFolder: [string] optional input that points to the folder where
%   the output file should be saved. If this is not specified, the output
%   file will be saved in the same folder as fname file.
%
% Outputs:
%   None 
%
% Example:
%   read_main_board_file([cd '\Input Data\HP_W10.2-5-20190820_155520.bin'], [cd '\Output Data']);
%       - This will read the file HP_W11.2-5-20190820_155520.bin, convert
%       it to HP_W11.2-5-20190820_155520.mat, which will be saved in the
%       subfolder 'Output Data' of the current directory

%% Handle the inputs

if(nargin > 1)
    if(~exist(outputFolder, 'dir'))
        error('outputFolder is not a valid path!');
    end
else
    outputFolder = fileparts(fname);
end

if(~exist(fname, 'file'))
    error('fname file does not exist!');
end

if(outputFolder(end)~=filesep)
    outputFolder = char(outputFolder);  % Handle windows copypath function
    outputFolder(end+1) = filesep;
end

outputFolder = char(outputFolder);  % Handle windows copypath function

%% Read the header info

% Read the header info
headerLength = 512;
fid = fopen(fname);
header = fread(fid, headerLength, 'uint8=>uint8');
fclose(fid);

% Pull out file information from header
deviceSerial        = header(1:16, :);
projectNum          = header(17:18, :);
hwRevMajor          = header(19:20, :);    
hwRevMinor          = header(21:22, :); 
HP_Serial           = header(23:24, :); 
devFirmwareVersion  = header(25:28, :);
BME_Cal             = header(29:61, :);
numSDBlocks         = header(62:65, :);
fileTime            = header(66:73, :);

% Convert from bytes into integer
deviceSerial        = double(typecast(deviceSerial(:),      	'uint32'));
projectNum          = double(typecast(projectNum(:),            'uint16'));
hwRevMajor          = double(typecast(hwRevMajor(:),            'uint16'));
hwRevMinor          = double(typecast(hwRevMinor(:),            'uint16'));
HP_Serial           = double(typecast(HP_Serial(:),             'uint16'));
devFirmwareVersion  = double(typecast(devFirmwareVersion(:),    'uint32'));
numSDBlocks         = double(typecast(numSDBlocks(:),           'uint32'));

% Convert the filetime 
% If Matlab >= R2018b (a?) can just use: fileTime_converted  = datetime(fileTime, 'ConvertFrom', 'ntfs');
% This may need a timezone correction; not sure.
fileTime = double(typecast(flipud(fileTime), 'uint64'));            % Flip since in reverse byte order
numTicks_1601_01_01_to_1900_01_01 = 94354848000000000;              % https://community.nintex.com/t5/Community-Blogs/Convert-date-time-format-to-Active-Directory-timestamp/ba-p/78047
fileTime_1900 = fileTime - numTicks_1601_01_01_to_1900_01_01;       % Get the time since 1901-01-01
timeZoneOffset = -java.util.Date().getTimezoneOffset()/60;          % System time zone offset
fileTime_1900 = fileTime_1900 + timeZoneOffset*60*60/100e-9;        % Remove the time zone error
numTicksPerDay = 24*60*60/100e-9;                                   % Define the number of 100 ns ticks per day
excelTime = fileTime_1900 / numTicksPerDay;                         % Excel time is in days
excelTime = excelTime + 2;                                          % Excel incorrectly has leap year first year; also need to get to Jan-0 (not Jan-1)
fileTime = datetime(excelTime, 'ConvertFrom', 'excel');             % Convert to readable file time

%% Save the header info

file_audio.deviceSerial        = deviceSerial;
file_audio.projectNum          = projectNum;
file_audio.hwRevMajor          = hwRevMajor;
file_audio.hwRevMinor          = hwRevMinor;
file_audio.HP_Serial           = HP_Serial;
file_audio.devFirmwareVersion  = devFirmwareVersion;
file_audio.numSDBlocks         = numSDBlocks;
file_audio.fileTime            = fileTime;

file_audio.matlabVersion       = 1.0;   % TODO: UPDATE THIS LINE OF CODE WHEN MAKE A CHANGE!!!

file_audio.fs_ast              = 4096;


%% Define constants based off the firmware revision

switch(file_audio.devFirmwareVersion)
    case 1
        num_bits_audio = 16;
        fs_audio = 46.875e3;
    case 2
        num_bits_audio = 16;
        fs_audio = 46.875e3;
    otherwise
        warning('Unrecognized firmware version! Using default number of bits and sample rate');
end

file_audio.numBits = num_bits_audio;
file_audio.fs = fs_audio;


%% Read the audio file

file_data = get_audio_board_file_helper(fname, file_audio.fs, file_audio.numBits);

if(isempty(file_data))
    % No data found in the file...
else
    % Organize the data
    file_audio.tt               = file_data.tt;
    file_audio.ch1              = file_data.ch1;
    file_audio.ch2              = file_data.ch2; 
    file_audio.ch3              = file_data.ch3;
    file_audio.ch4              = file_data.ch4; 
    file_audio.tt_blocks        = file_data.tt_blocks;
    file_audio.tt               = file_data.tt;
    file_audio.startTime        = file_data.startTime;
    file_audio.stopTime         = file_data.stopTime;
    file_audio.firstSampTime    = file_data.firstSampTime;
    file_audio.firstSampRaw     = file_data.firstSampRaw;
    file_audio.secondSampRaw    = file_data.secondSampRaw;
end

%% Save the data

[~, outname, ~] = fileparts(fname);
fnameOut = [outputFolder, char(outname), '.mat'];
save(fnameOut, 'file_audio');

end

function [file_audio] = get_audio_board_file_helper(fname, fs, numBits)
%%

%% Define DEBUG variables

B_DEBUG_PLOT = 0;   % Debug plots: drop sample analysis


%% Define constants

headerLength = 512;         % Number of bytes per block
fs_ast = 4096;              % AST frequency
% buffLength = 4096;        % Number of uint32 per half buffer
adc_half_buffer = 5504;     % Number of bytes per half buffer
tt_block_length = 128;      % Length (uint32) of sample time block

%% Read the data

% Move to the correct place on the SD card and pull data
fid = fopen(fname);
fseek(fid, headerLength, 'bof');
raw = fread(fid, Inf, 'uint32', 0, 'l'); % Need to divide by 4 because reading as uint32 and not uint8
fclose(fid);


%% Remove the padded blocks at the end of the file

raw = remove_padded_data_helper(raw);

if(isempty(raw))
    file_audio = [];
    return;
end

%% Get the sync time values

% Get the block with the time values and remove it from normal data stream
syncTime = raw(end-512/4+1:end);
raw(end-512/4+1:end) = [];

% Get the different time values
startAST    = syncTime(1);  % Time when gets switch interrupt
stopAST     = syncTime(2);  % TIme when gets switch interrupt
firstBuff   = syncTime(3);  % Time when first buffer complete plus AST sync delay
secondBuff  = syncTime(4);  % x

% Calculate the first sample time
delay_time = 0;                     % Use zero since imped sensors have to wait this delay as well (vs say 200 us for AST)
if(numBits==16)
    numSamps = adc_half_buffer/2;   % 2 uint32 per 4 channels = 1 sample
else
    numSamps = adc_half_buffer/4;   % 4 uint32 per 4 channels = 1 sample
end
buffTime = numSamps/fs;
totTime = buffTime + delay_time;

% Convert the AST timer ticks to seconds
file_audio.startTime = startAST/fs_ast;
file_audio.stopTime = stopAST/fs_ast;
file_audio.firstSampTime = firstBuff/fs_ast - totTime;
file_audio.firstSampRaw = firstBuff/fs_ast;
file_audio.secondSampRaw = secondBuff/fs_ast;


%% Split the data into ADC values and ssample times

% Organize into packets
num_uint32_per_write = adc_half_buffer + tt_block_length;  % 32 data blocks and 1 time block
numPackets = length(raw)/num_uint32_per_write;
packets = reshape(raw, num_uint32_per_write, numPackets);

% Get the raw ADC data
raw_data = packets(1:adc_half_buffer, :);
raw_data = reshape(raw_data, size(raw_data, 1)*size(raw_data, 2), 1);

% Get the raw time data
time_blocks_packets = packets(adc_half_buffer+1:end, :);
time_blocks = reshape(time_blocks_packets, size(time_blocks_packets, 1)*size(time_blocks_packets, 2), 1);
time_blocks(time_blocks==0) = [];
time_blocks = unwrap_count_vector_helper(time_blocks);
tt_blocks = convert_ast_count_to_time_helper(time_blocks, fs_ast);


%% Convert the raw ADC values to voltage

% Split the data into channels
[ch1, ch2, ch3, ch4] = convert_sd_raw_audio_to_decimal_helper(raw_data, numBits);

% Convert raw ADC to voltage
ch1_volt = convert_audio_decimal_to_voltage_helper(ch1, numBits);
ch2_volt = convert_audio_decimal_to_voltage_helper(ch2, numBits);
ch3_volt = convert_audio_decimal_to_voltage_helper(ch3, numBits);
ch4_volt = convert_audio_decimal_to_voltage_helper(ch4, numBits);


%% Construct a time vector

tt = 0:1/fs:(length(ch1_volt)-1)/fs;
tt = tt(:);
tt_corrected = tt + (file_audio.firstSampTime - file_audio.startTime);
file_audio.tt = tt_corrected;

% % TODO: need to handle the dropped sample issue AND test
% d_blocks = diff(time_blocks);
% if(numBits==16)
%     ind = find(d_blocks ~= 240 & d_blocks~=241);    % Find case where drop sample
%     if(~isempty(ind))
%         error('Dropped Samples!');
%     end
% else
%     % TODO: add code for dropped samples 
%     error('Code for 24 bits needed!');
% end


%% Save the data to the output struct

file_audio.tt_blocks = tt_blocks;   % Sample time per block
file_audio.tt = tt_corrected;       % 'Artificial sample time
file_audio.ch1 = ch1_volt;
file_audio.ch2 = ch2_volt;
file_audio.ch3 = ch3_volt;
file_audio.ch4 = ch4_volt;


%% Debug plotting regarding drop samples

if(B_DEBUG_PLOT)

    figure(99); plot(diff(time_blocks(1:end)));

    plot_tt = tt_blocks./60;
    
    figure(100);hold on;plot(plot_tt(1:end-1),diff(time_blocks(1:end)));xlabel('Time (min)');ylabel('\DeltaAST');title('\DeltaAST for Audio');
    figure(101);hold on;plot(plot_tt(1:end-1),diff(tt_blocks(1:end)));xlabel('Time (min)');ylabel('\Deltat (sec)');title('\DeltaTime for Audio');
    figure(102);hold on;histogram(diff(time_blocks(1:end)));xlabel('\DeltaAST');ylabel('Number of Blocks');title('\DeltaAST Values for Audio');
    figure(100);hold on;plot(plot_tt(1:end-1),diff(time_blocks(1:end)), '.');xlabel('Time (min)');ylabel('\DeltaAST');title('\DeltaAST for Audio');
    figure(101);hold on;plot(plot_tt(1:end-1),diff(tt_blocks(1:end)), '.');xlabel('Time (min)');ylabel('\Deltat (sec)');title('\DeltaTime for Audio');
end

end

function [ch1, ch2, ch3, ch4] = convert_sd_raw_audio_to_decimal_helper(raw, numBits)


%% Split the channels

if(numBits==24)
    % Remove unnecessary data
    bin = de2bi(raw, 32, 'left-msb');
    bin(:, 1) = [];
    bin(:, end+1) = 0;
    dec = bi2de(fliplr(bin));
    
%     raws = bitsll(raw, 1);        % bit shift by 1 for offset of sent data   
%     bin = de2bi(raws, 33);
%     bin(:, 1:33-numBits) = [];
%     dec2 = bi2de(bin);

    
    % Split the channels
    ch1 = dec(1:4:end);
    ch2 = dec(2:4:end);
    ch3 = dec(3:4:end);
    ch4 = dec(4:4:end);
else
%     % Divide the upper and lower bytes
%     bin = de2bi(raw, 32, 'left-msb');
%     bin12 = bin(:, 1:16);
%     bin34 = bin(:, 17:32);
%     
%     % Reconstruct the data
%     dec12 = bi2de(fliplr(bin12));
%     dec34 = bi2de(fliplr(bin34));
%     
%     
%     % Split the channels
%     ch1 = dec12(1:2:end, :);
%     ch2 = dec12(2:2:end, :);
%     ch3 = dec34(1:2:end, :);
%     ch4 = dec34(2:2:end, :);

    % Divide the upper and lower bytes
    bin = de2bi(raw, 32, 'left-msb');
    bin13 = bin(:, 1:16);
    bin24 = bin(:, 17:32);
    
    % Reconstruct the data
    bin13 = bi2de(fliplr(bin13));
    bin24 = bi2de(fliplr(bin24));
    
    
    % Split the channels
    ch1 = bin13(1:2:end, :);
    ch3 = bin13(2:2:end, :);
    ch2 = bin24(1:2:end, :);
    ch4 = bin24(2:2:end, :);   
end


end

function [voltage] = convert_audio_decimal_to_voltage_helper(data_dec, numBitsData)

%% Preprocess data

if(numBitsData==24)
    % NOTE: LSB is first
    data_bin = de2bi(data_dec, 32, 'left-msb');

    % Extend the sign bit
    signBit = data_bin(:, 1);
    signExtend = signBit.*ones(length(signBit), 32 - numBitsData);
    twos = [signExtend, data_bin(:, 1:numBitsData)]; 
else
    % NOTE: LSB is first
    data_bin = de2bi(data_dec, 16, 'left-msb');

    % Extend the sign bit
    signBit = data_bin(:, 1);
    signExtend = signBit.*ones(length(signBit), 32 - numBitsData);
    twos = [signExtend, data_bin];
end

% Convert to string
twos = char(twos+48);
    
%% Convert to voltage

data = typecast(uint32(bin2dec(twos)),'int32'); 
voltage = (double(data) .* 2 .* 4.5 .* sqrt(2) ./ (2.^numBitsData)) + 1.5;

end

function [act_data] = remove_padded_data_helper(raw)
%% REMOVE_PADDED_DATA_HELPER
% Removes the zero padded data at the end of the file, which was added to
% make the file length a multiple of the wearable data file length so that
% USB code host app could be directly used.
%
% The function finds the blocks of all zeros at the end of the file and
% removes them. In all probability, the time block will be non-zero values
% for both the start and end times. The time block at the end will not be
% removed. 

%% Define constants

block_length = 512/4;                       % Block length (uint32)
padded_block = zeros(1, block_length);      % Padded block pattern


%% Remove the zero padding at the end

% Reshape into blocks
blocks = reshape(raw, block_length, length(raw)/block_length);

% Find blocks that match the padded block pattern
[~, index] = ismember(blocks', padded_block, 'rows');
padded_cols = find(index==1);
padded_cols = padded_cols(:);

% Only look at the blocks at the end
if(sum(ismember(padded_cols, size(blocks, 2))))  % At least one at the end
    % If there are more than one block, only include the ones at the end
    if(length(padded_cols)>1)
        diff_padded_cols = diff(padded_cols);
        ind = find(diff_padded_cols>1, 1, 'last') + 1;
        if(~isempty(ind))
            % There are other padded-patern columns, so only remove ones at
            % end
            padded_cols = padded_cols(ind:end);
        end
    end
    
    % Remove the padded blocks
    blocks_rem = blocks;
    blocks_rem(:, padded_cols) = [];
    blocks_rem = reshape(blocks_rem, size(blocks_rem, 1)*size(blocks_rem, 2), 1);
    act_data = blocks_rem;
else
    % There are no padded blocks; return entire data set
    act_data = raw;
end


end

function [new_count] = unwrap_count_vector_helper(count)
%% Define constants

maxCount = 2^32 - 1;

%% Unwrap the counter value

% Example:
% count = [... 253, 0, 4, ...];
% dx = [-253 4];
% inds = [1];
% The counter counts to max value (255) and then rollovers. So from 253, it
% counts 254, 255, 0. So there are three ticks that have occurred.
% dTick = 3;
% If there was no rollover condition, it would have counted 253, 254, 255,
% 256, which is 253 + 3 or: 
% new_count(inds(ii)) + dTick + new_count(inds(ii)+1:inds(ii+1))
% This is more clear for the 4 case. 4 without rollover would have been
% 260. This is equal to 253 + 3 + 4 from the equation above.

dx = diff(count);
inds = find(dx<-maxCount/2);
new_count = count;
for ii=1:length(inds)
    % Find the number of ticks difference
    dTick = maxCount - count(inds(ii)) + ...    % Num ticks until max
            count(inds(ii)+1) + 1;              % Num ticks after rollover (plus one to get the 0 value)
    
    % Account for the tick difference
    if(ii<length(inds))
        new_count(inds(ii)+1:inds(ii+1)) = new_count(inds(ii)) + ...                % New offset
                                           dTick + ...                              % Num ticks to get offset from new offset
                                           new_count(inds(ii)+1:inds(ii+1));        % Current value
    else
        new_count(inds(ii)+1:end) = new_count(inds(ii)) + ...                       % New offset
                                    dTick + ...                                     % Num ticks to get offset from new offset
                                    new_count(inds(ii)+1:end);                      % Current value
    end
end


end

function [tt] = convert_ast_count_to_time_helper(count, fs_ast)
%%

%% Convert from AST counter value to time

tt = count ./ fs_ast;     % 32768 / 4

end





















