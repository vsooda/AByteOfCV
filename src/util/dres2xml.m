function trackXml() 
	load('../cache/gt_dp_nms10.mat');
	id_num = max(dres_dp_nms.id);

    bbox = [];
    
	for i = 1 : id_num, 
		f = find(dres_dp_nms.id == i);
		f = f(end:-1:1);
        
        bb.x = dres_dp_nms.x(f);
		bb.y = dres_dp_nms.y(f);
		bb.h = dres_dp_nms.h(f);
		bb.w = dres_dp_nms.w(f);
        frame = dres_dp_nms.fr(f);
        start_frame = min(frame);
        bb.st = start_frame;
        bb.num = max(frame) - start_frame + 1;
        
        if isempty(bbox),
            bbox = bb;
        else
           bbox = [bbox bb];
        end
	end



	docNode = com.mathworks.xml.XMLUtils.createDocument('opencv_storage');
	docRootNode = docNode.getDocumentElement;
	%docRootNode.setAttribute('name','PETS-S3-MF1');
	
	infolistNode = docNode.createElement('infolist');
	
	for i = 1 : id_num,
		info_name = sprintf('info_%04d', i);
		infoNode = docNode.createElement(info_name);
		info = sprintf('%d %d %d', i, bbox(i).st, bbox(i).num);
		infoNode.appendChild(docNode.createTextNode(info));
        infolistNode.appendChild(infoNode);
	end
	

	docRootNode.appendChild(infolistNode);


	tracklistNode = docNode.createElement('tracklist');
	for i = 1 : id_num,
		track_name = sprintf('track_%04d', i);
		frameNode = docNode.createElement(track_name);
		cnt = sprintf('%03d', i);
	%	frameNode.setAttribute('id', num2str(i));
	%	frameNode.setAttribute('cnt', num2str(bbox(i).num));
		for j = 1 : bbox(i).num,
			boxItem_name = sprintf('box_%04d', j);
			objectNode = docNode.createElement(boxItem_name);

			xNode = docNode.createElement('x');
			xNode.appendChild(docNode.createTextNode(num2str(bbox(i).x(j))));
			objectNode.appendChild(xNode);

			yNode = docNode.createElement('y');
			yNode.appendChild(docNode.createTextNode(num2str(bbox(i).y(j))));
			objectNode.appendChild(yNode);

			hNode = docNode.createElement('h');
			hNode.appendChild(docNode.createTextNode(num2str(bbox(i).h(j))));
			objectNode.appendChild(hNode);

			wNode = docNode.createElement('w');
			wNode.appendChild(docNode.createTextNode(num2str(bbox(i).w(j))));
			objectNode.appendChild(wNode);

		%	objectNode.setAttribute('id', num2str(j));
			%boxNode = docNode.createElement('box');
	%		xywh = sprintf('%f %f %f %f', bbox(i).x(j), bbox(i).y(j), bbox(i).w(j), bbox(i).h(j));
	%		objectNode.appendChild(docNode.createTextNode(xywh));
%		    boxNode.setAttribute('h', num2str(bbox(i).h(j)));
%		    boxNode.setAttribute('w', num2str(bbox(i).w(j)));
%		    boxNode.setAttribute('xc', num2str(bbox(i).x(j)));
%		    boxNode.setAttribute('yc', num2str(bbox(i).y(j)));
			%objectNode.appendChild(boxNode);
			frameNode.appendChild(objectNode);
        end
		tracklistNode.appendChild(frameNode);
		%showboxes(im, bbox);
	%	disp('please any key to continue...');
	%	pause;
	%	disp('continue...');
	end
	docRootNode.appendChild(tracklistNode);
	xmlwrite('track_gt.xml',docNode);
end
