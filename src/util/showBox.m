function showBox(im, box) 

imshow(im, 'Border', 'tight');
hold on

m = size(box);
for i = 1 : m,
	bleft = box(i, 1);
	bright = box(i, 3);
	btop = box(i, 2);
	bbottom = box(i, 4);
	col = [.1 .2 .3];
	line([bleft bleft bright bright bleft], [btop bbottom bbottom btop btop], 'color', col, 'linewidth', 1);
	text(bleft, btop-5, num2str(box(i, 5)), 'fontsize', 30, 'color', 'b');
end

end

