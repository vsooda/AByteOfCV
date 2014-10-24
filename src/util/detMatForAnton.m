function detMatForAnton() 
	%% xi yi ht wd sc
	frameNum = 795;
	load('cache/dres_nms.mat');
	boxes = dres2bboxes(dres_dp_nms, frameNum);
	size(boxes);
	%boxes(795).bbox
%	for i = 1 : frameNum,
%		bboxes = boxes(i).bbox;
%		bx = bboxes(1)
%	end
%	boxes(795).bbox
%	boxes(795).bbox(:, 1)
	for i = 1 : frameNum,
		bbox = boxes(i).bbox
		ltopx = bbox(:, 1)';
		ltopy = bbox(:, 2)';
		rBottomx = bbox(:, 3)';
		rBottomy = bbox(:, 4)';
		sc = bbox(:, 5)';
		detections(i).xi = ltopx + (rBottomx - ltopx) / 2;
		detections(i).yi = ltopy + (rBottomy - ltopy) / 2;
		detections(i).ht = rBottomy - ltopy;
		detections(i).wd = rBottomx - ltopx;
		detections(i).sc = sc;
		detections(i)
	end
	save('cache/det.mat', 'detections');
end
