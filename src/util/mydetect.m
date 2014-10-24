function mydetect(vid_path)
% vid_path = 'data/seq03-img-left/';
thresh = -2;              %% threshod on SVM response in human detection, we'll have more detections by decreasing it.

tmp = load ('3rd_party/voc-release3.1/INRIA/inria_final.mat');  %% load the model for human. This can be changed to any of those 20 objects in PASCAL competition.
model= tmp.model;
clear tmp

  im = imread(vid_path);
%  im = imresize(im,2);                %% double the image size to detect small objects.
  
  boxes = detect(im, model, thresh);  %% running the detector
  bbox =  getboxes(model, boxes);
  
%  pause;
  box = nms(bbox, 0.5);    %% running non-max-suppression to suppress overlaping weak detections.
  showBox(im, box);
  display('enter any key to continue...');
  pause;
end
