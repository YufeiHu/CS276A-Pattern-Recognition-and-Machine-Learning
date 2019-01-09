function [index] = findClass(class_dic, class_name)

    num_class = size(class_dic, 2);
    for i = 1 : num_class
        if(strcmp(class_dic{i}, class_name))
            break
        end
    end
    index = i;

end