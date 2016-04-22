function getImPaths(list_file_path)    
    
    local imPaths = {}
    for line in io.lines(list_file_path) do
        words = line:split(' ')
        im_Path, label = words[1], words[2]
        label = tonumber(label)
        if imPaths[label] == nil then
            imPaths[label] = {}
        end        
        table.insert(imPaths[label], im_Path)
    end
    
    return imPaths    
end