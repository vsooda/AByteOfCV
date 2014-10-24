function parseDetMat() 
	load('1.mat');
	dres.x = [];
	dres.y = [];
	dres.w = [];
	dres.h = [];
	dres.r = [];
	dres.fr = [];
	frameNum = 795;
	for i = 1 : frameNum,
		len = length(detections(i).x);
		for j = 1 : len,
			dres.x = [dres.x; detections(i).x(j)];
			dres.y = [dres.y; detections(i).y(j)];
			dres.h = [dres.h; detections(i).h(j)];
			dres.w = [dres.w; detections(i).w(j)];
			dres.r = [dres.r; detections(i).sc(j)];
			dres.fr = [dres.fr; i];
		end
	end
	save('gt.mat', 'dres');
end
