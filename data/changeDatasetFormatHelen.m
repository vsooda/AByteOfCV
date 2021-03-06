%%helen
clear; clc; close all;
addpath functions


%% select database and load bb initializations
load bounding_boxes_helen_testset

%% Select image
fout = fopen('helen_test_bb_d.txt', 'w');

bbs = cell2mat(bounding_boxes);
len = length(bounding_boxes)

for (i = 1 : len) 
	fprintf(fout, '%s\n', bbs(i).imgName);	
	%bb = bbs(i).bb_ground_truth;
    bb = bbs(i).bb_detector;
	%fprintf(fout, '%f %f %f %f', bb(2), bb(1), bb(4), bb(3));
	bb = uint32(bb);

	%fprintf(fout, '%d %d %d %d\n', bb(2), bb(1), bb(4), bb(3));
    fprintf(fout, '%d %d %d %d\n', bb(1), bb(2), bb(3), bb(4));
end

fclose(fout);


