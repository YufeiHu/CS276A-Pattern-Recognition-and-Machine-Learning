function [bounding_boxes_adjust, scores_adjust] = adjust_bounding_boxes(bounding_boxes, scores, image_width, image_height)

    N = size(bounding_boxes, 1);
    delete_index = [];
    
    for i = 1 : N
        x = bounding_boxes(i, 1);
        y = bounding_boxes(i, 2);
        width = bounding_boxes(i, 3);
        height = bounding_boxes(i, 4);
        
        if((x>image_width) || (y>image_height))
            delete_index = [delete_index i];
            continue
        end
        
        
        if(x<0)
            bounding_boxes(i, 1) = 0;
            x = 0;
        end
        if(y<0)
            bounding_boxes(i, 2) = 0;
            y = 0;
        end
        
        
        if((width==0) || (height==0))
            delete_index = [delete_index i];
            continue
        end
        
        
        if((x+width)>image_width)
            bounding_boxes(i, 3) = image_width - x;
        end
        if((y+height)>image_height)
            bounding_boxes(i, 4) = image_height - y;
        end
        
    end
    
    bounding_boxes(delete_index, :) = [];
    scores(delete_index, :) = [];
    
    bounding_boxes_adjust = bounding_boxes;
    scores_adjust = scores;
    
end