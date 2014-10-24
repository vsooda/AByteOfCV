function myshowFrame() 
framePause = 1;
col = [.1 .2 .3];
frameNum = 795;
%load('cache/seq03-img-left_detec_res.mat');
%boxes = bboxes;
%load('cache/seq03-img-left_graph_res.mat');
%dres.id = dres.r;
%boxes = dres2bboxes(dres, frameNum);
load('../cache/dres_nms.mat');
boxes = dres2bboxes(dres_dp_nms, frameNum);
for i = 1:795,
	picNum = i - 1;
	filename = ['data/seq03-img-left/' sprintf('meet_%03d.jpg', picNum)];
	im = imread(filename);
%	im = imresize(im, 2);
	imshow(im);
	hold on
	text(20, 50, sprintf('%d', picNum), 'FontSize', 30);
	
	box = boxes(i).bbox;
	showBox(im, box);
	pause(framePause);	
end
end
