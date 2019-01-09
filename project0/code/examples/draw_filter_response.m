% Draw the filter responses of the first layer


function [ ] = draw_filter_response(model, net)


index = 'images/';
for image_index = 1 : 10
    image_path = strcat('images/', num2str(image_index), '.png');
    img = imread(image_path);
    figure;
    imshow(img);
    img = single(img) - model.net.averageImage;
    res = vl_simplenn(net, img);
    responses = res(2).x;
    
    figure;
    for i = 1 : 32
        subplot(4,8,i);
        image(responses(:,:,i),'CDataMapping','scaled');
    end
    axis tight;
    axis off;
    set(findobj(gcf, 'type','axes'), 'Visible','off')
    daspect([1 1 1]);
end


end