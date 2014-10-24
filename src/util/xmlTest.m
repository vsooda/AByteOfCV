function xmlTest(tempname) 
%	docNode = com.mathworks.xml.XMLUtils.createDocument('dataset');
%	docRootNode = docNode.getDocumentElement;
%	docRootNode.setAttribute('name','PETS-S3-MF1');
%	for i=1:20
%	   thisElement = docNode.createElement('child_node'); 
%   	   thisElement.appendChild(docNode.createTextNode(sprintf('%i',i)));
%	   docRootNode.appendChild(thisElement);
%	end
%	docNode.appendChild(docNode.createComment('this is a comment'));
%	xmlFileName = [tempname,'.xml'];
%	xmlwrite(xmlFileName,docNode);
%	type(xmlFileName);
	xmlDoc = xmlread(fullfile('2.xml'));
	items = xmlDoc.getElementsByTagName('frame');
	for i =0:99 
		it = items.item(i)
		it.setAttribute('number', num2str(i+650));
	end
	xmlwrite('2.xml', xmlDoc);
end

