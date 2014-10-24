function videoDetectXml() 
	load('cache/dres_nms.mat');
	path = 'demo/img/';
	det_path = 'demo/det/';
	docNode = com.mathworks.xml.XMLUtils.createDocument('dataset');
	docRootNode = docNode.getDocumentElement;
	docRootNode.setAttribute('name','PETS-S3-MF1');
	for i = 650 : 749,
		frameNode = docNode.createElement('frame');
		cnt = sprintf('%04d', i)
	%	frameRootNode = frameNode.getDocumentElement;
		frameNode.setAttribute('number', num2str(i));
		objlistNode = docNode.createElement('objectlist');
	%	objlistRootNode = objlistNode.getDocumentElement;
		filename = [path, 'frame_', cnt, '.jpg'];
		det_filename = [path, 'frame_', cnt, '.jpg'];
		im = imread(filename);
		%imshow(im), pause;
		bbox = process(im, model, -0.5)
		m = size(bbox, 1)
		for j = 1 : m,
			ltop_x = bbox(j, 1);
			ltop_y = bbox(j, 2);
			rbottom_x = bbox(j, 3);
			rbottom_y = bbox(j, 4);
			h = rbottom_y - ltop_y
			w = rbottom_x - ltop_x
			xc = (ltop_x + rbottom_x) / 2
			yc = (ltop_y + rbottom_y) / 2

			objectNode = docNode.createElement('object');
	%		objectRootNode = objectNode.getDocumentElement;
			objectNode.setAttribute('confidence', num2str(bbox(j, 6)));
			boxNode = docNode.createElement('box');
	%		boxRootNode = boxNode.getDocumentElement;
		    boxNode.setAttribute('h', num2str(h));
		    boxNode.setAttribute('w', num2str(w));
		    boxNode.setAttribute('xc', num2str(xc));
		    boxNode.setAttribute('yc', num2str(yc));
			objectNode.appendChild(boxNode);
			objlistNode.appendChild(objectNode);
		end
		frameNode.appendChild(objlistNode);
		docRootNode.appendChild(frameNode);
		%showboxes(im, bbox);
	%	disp('please any key to continue...');
	%	pause;
	%	disp('continue...');
	end
	xmlwrite('2.xml',docNode);
end
