from Const import output_geometry_file_separator as gfs


class GeometryReader:
    def __init__(self, file_name):
        self.txt = {gfs[0]: [], gfs[1]: [], gfs[2]: []}
        self.gridpoints = {}
        self.vertices = {}
        self.faces = {}
        self.blocks = {}
        with open(file_name, 'rb') as f:
            lines = f.readlines()
            separator_count = 0
            record = False
            decode_spec = "utf-8"
            for i, line in enumerate(lines):
                line = line.decode(decode_spec)
                line = line.replace('\n', '')
                if '*' in line and record:
                    record = False
                    separator_count += 1
                    if separator_count > 2:
                        break
                if record:
                    self.txt[gfs[separator_count]].append(line)
                if gfs[separator_count] in line:
                    record = True

    def get_block_vertices(self):
        block_vertices = []
        gp_read_count = 0
        # For each block
        for i, block_txt in enumerate(self.txt[gfs[2]]):
            vertices = []
            while len(vertices) < 8:
                gp_txt = self.txt[gfs[0]][gp_read_count].split()  # Get the split information of one single line of grid point description
                vertex = [float(j) for j in gp_txt[2:5]]  # Select the coordinates and convert the scientific notation
                vertices.append(vertex)
                gp_read_count += 1
            vertices = transform_vertex_3DEC(vertices)
            block_vertices.append(vertices)

        return block_vertices


def transform_vertex_3DEC(vertices):
    temp = vertices.copy()
    temp[1] = vertices[4]
    temp[2] = vertices[5]
    temp[3] = vertices[1]
    temp[4] = vertices[3]
    temp[5] = vertices[6]
    temp[6] = vertices[7]
    temp[7] = vertices[2]
    return temp


if __name__ == '__main__':
    filename = 'text_function2.txt'
    a = GeometryReader(filename)
    for key in a.txt:
        sample = a.txt[key][0]
        sample_split = sample.split()
        print(sample_split)
        print(sample_split[3])
        print(float(sample_split[3]))
    for block_verts in a.get_block_vertices():
        print(block_verts)

