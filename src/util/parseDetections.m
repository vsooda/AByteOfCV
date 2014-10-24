function parseDetections(detfile,frames)
%%parseDetections('det.xml', 795);
xDoc=xmlread(detfile);
allFrames=xDoc.getElementsByTagName('frame');
F=allFrames.getLength;

for t=1:F, 
    frame=str2double(allFrames.item(t-1).getAttribute('number'));

    objects=allFrames.item(t-1).getElementsByTagName('object');
    
    Nt=objects.getLength;
    nboxes=Nt; % how many detections in current frame
    x=zeros(1,nboxes);
    y=zeros(1,nboxes);
    h=zeros(1,nboxes);
    w=zeros(1,nboxes);
    
    scores=zeros(1,nboxes);
    
    for i=0:Nt-1
        % score
        boxid=i+1;
        scores(boxid)=str2double(objects.item(i).getAttribute('confidence'));
        box=objects.item(i).getElementsByTagName('box');

        % box extent
        h(boxid) = str2double(box.item(0).getAttribute('h'));
        w(boxid) = str2double(box.item(0).getAttribute('w'));

        % foot position
        x(boxid) = str2double(box.item(0).getAttribute('xc'));
        y(boxid) = str2double(box.item(0).getAttribute('yc'));

    end
    
	x = x - w / 2;
	y = y - h / 2;
    detections(t).x=x;
    detections(t).y=y;
    detections(t).h=h;
    detections(t).w = w;
    detections(t).sc=scores;


end
    

% save detections in a .mat file
save('1.mat','detections');

end
