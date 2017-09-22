import matplotlib.patches as patches
class lec4:
    def __init__(self):
        self.saved_box = []

    def show_box(self, ax, group_index, item_index, size, color):
        group_i = (group_index) % 2
        group_j = group_index // 2
        group_x = group_i * 50 + 2
        group_y = (2 - group_j) * 10 + 1
        box_width = 3.75
        box_height = 2.6
        item_i = (item_index) % 12
        item_j = item_index // 12
        item_x = item_i * box_width + 2
        item_y = (2 - item_j) * box_height + 0


        if group_index*24+item_index<len(self.saved_box):
            box = self.saved_box[group_index*24+item_index]
            box.set_width(size)
            box.set_height(size)
            box.set_xy((group_x + item_x - size / 2, group_y + item_y - size / 2))
            box.set_facecolor(color)
        else:
            rect = patches.Rectangle(xy=(group_x + item_x - size / 2, group_y + item_y - size / 2), width=size, height=size,
                                     facecolor=color)
            self.saved_box.append( ax.add_patch(rect) )

    def show_name(self, ax, item_index, text):
        base_offset_x = 1
        base_offset_y = 25
        base_right_offset_x = 50
        box_width = 3.75
        box_height = 2.6
        item_i = (item_index) % 12
        item_j = item_index // 12
        item_x = item_i * box_width + 2
        item_y = (2 - item_j) * box_height + 0
        ax.text(base_offset_x+item_x, base_offset_y+item_y, text, rotation=90, fontsize='larger', va='bottom')
        ax.text(base_right_offset_x+base_offset_x+item_x, base_offset_y+item_y, text, rotation=90, fontsize='larger', va='bottom')

    def show_background(self, ax):
        ax.add_patch(patches.Rectangle(xy=(2, 21), width=46, height=8, facecolor='gray'))
        ax.add_patch(patches.Rectangle(xy=(2, 11), width=46, height=8, facecolor='gray'))
        ax.add_patch(patches.Rectangle(xy=(2, 1), width=46, height=8, facecolor='gray'))
        ax.add_patch(patches.Rectangle(xy=(52, 21), width=46, height=8, facecolor='gray'))
        ax.add_patch(patches.Rectangle(xy=(52, 11), width=46, height=8, facecolor='gray'))
        ax.add_patch(patches.Rectangle(xy=(52, 1), width=46, height=8, facecolor='gray'))
