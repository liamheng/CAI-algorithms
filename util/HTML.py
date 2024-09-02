import os.path as osp
from glob import glob

class Item(object):
    def __init__(self, name, parent_path, visual_list):
        self.name = name
        self.parent_path = parent_path
        self.visual_list = visual_list
    
    def generate_html_elememt(self):
        html = '\t<div class=item>\n'

        # name of item
        html += '\t\t<div class=name> %s </div>\n' % self.name
        
        # visuals of item
        table = '\t\t<table>\n\t\t\t<tr>\n'
        for visual in self.visual_list:
            path = osp.join(self.parent_path, '%s_%s.png' % (self.name, visual))
            table += '\t\t\t\t<td><img src=%s width="200"></td>\n' % path
        table += '\t\t\t</tr></table>\n'
        html += table

        html += '\t</div>'

        return html

class HTML(object):
    def __init__(self, visual_list,  root_path, save_path='index.html'):
        self.items = []

        # get name list 
        with open(osp.join(root_path, 'names.txt'), 'r') as f:
            name_list = f.readlines()
        for name in name_list:
            self.items.append(Item(name.strip(), './image', visual_list))
        self.save_path = save_path
        self.name_list = name_list
            
    def write2html(self):
        html = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<title>Page Title</title>\n</head>\n<body>\n'
        for item in self.items:
            html += item.generate_html_elememt()
        html += '</body>\n</html>'
        with open(self.save_path, 'w') as f:
            f.write(html)
            print("[Write html to %s]" % self.save_path)