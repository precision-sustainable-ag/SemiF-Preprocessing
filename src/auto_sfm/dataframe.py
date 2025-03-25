class DataFrame:
    """A custom class which emulates pandas DataFrame
       for a limited functionality. The class needs
       a list of dictionaries as it's input. Each element
       in the list corresponds to a row, and each key in the
       dictionary correcponds to a column.
    """
    def __init__(self, content_dict, primary_key, sep=","):
        
        self.content_dict = content_dict
        self.primary_key = primary_key
        self.sep = sep
        self.content = self.parse_dict(content_dict)
        self.indices = self.build_key_indices()

    def parse_dict(self, content):

        # Get a list of all unique keys
        columns = set()
        for row in content:
            columns = columns.union(set(list(row.keys())))
        columns = list(columns)

        content_string = [self.sep.join(columns)]
        for row in content:
            row_content = []
            for key in columns:
                element = str(row.get(key, ""))
                row_content.append(element)
            parsed_row = self.sep.join(row_content)
            content_string.append(parsed_row)
        content_string = "\n".join(content_string)

        return content_string

    def build_key_indices(self):

        indices = {row[self.primary_key]: i for i, row in enumerate(self.content_dict)}
        return indices

    def to_csv(self, filepath, *args, **kwargs):
        """Writes the contents to a CSV file

        Args:
            filepath ([type]): Path to the CSV file
            *args: Arguments to make the function compatible with the Pandas API
        """

        with open(filepath, "w") as f:
            f.write(self.content)

    def __len__(self):
        return len(self.content_dict)

    def retrieve(self, key_value, field):

        index = self.indices[key_value]

        return self.content_dict[index].get(field, "")


def read_csv(path, sep=","):

    with open(path, "r") as f:
        content = f.read()

    rows = content.split("\n")
    cols = [row.split(sep) for row in rows]

    # Make to a compatible structure
