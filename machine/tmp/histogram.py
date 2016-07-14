    def graph3D(self):
        """Graph histogram.

        Args:
            histogram_list: List for histogram
            category: Category (label) for data in list
            bins: Number of bins to use

        Returns:
            None

        """
        # Initialize key variables
        directory = '/home/peter/Downloads'
        bins = self.bins()
        categories = []
        lines2plot = []

        # Create the histogram plot
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')

        # Random colors for each plot
        prop_iter = iter(plt.rcParams['axes.prop_cycle'])

        # Loop through data
        for category in self.x_y.keys():
            # Get key data for creating histogram
            x_array = np.array(self.x_y[category][0])
            y_array = np.array(self.x_y[category][1])
            (hist, xedges, yedges) = np.histogram2d(
                x_array, y_array, bins=bins)

            # Number of boxes
            elements = (len(xedges) - 1) * (len(yedges) - 1)
            (xpos, ypos) = np.meshgrid(
                xedges[:-1] + 0.25, yedges[:-1] + 0.25)

            # x and y coordinates of the bars
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros(elements)

            # Lengths of the bars on relevant axes
            dx_length = 1.0 * np.ones_like(zpos)
            dy_length = dx_length.copy()
            dz_length = hist.flatten()

            # Append category name
            categories.append(category.capitalize())

            # Chart line
            lines2plot = axes.bar3d(
                xpos, ypos, zpos,
                dx_length, dy_length, dz_length,
                alpha=0.5,
                zsort='average',
                color=next(prop_iter)['color'],
                label=category.capitalize())


        """

        # Put ticks only on bottom and left
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('bottom')
        axes.zaxis.set_ticks_position('bottom')

        # Set X axis ticks
        major_ticks = np.arange(0, bins, 1)
        axes.set_xticks(major_ticks)

        # Set y axis ticks
        major_ticks = np.arange(0, bins, 1)
        axes.set_yticks(major_ticks)

        # Set z axis ticks
        major_ticks = np.arange(0, max(dz_length), 5)
        axes.set_zticks(major_ticks)

        """

        # Add legend
        # axes.legend(lines, categories)
        # plt.legend()

        # Add Main Title
        fig.suptitle(
            'Height and Handspan Histogram',
            horizontalalignment='center',
            fontsize=10)

        # Add grid, axis labels
        axes.grid(True)
        axes.set_ylabel('Handspans')
        axes.set_xlabel('Heights')
        axes.set_zlabel('Count')

        # Adjust bottom
        fig.subplots_adjust(left=0.2, bottom=0.2)

        # Create image
        graph_filename = ('%s/homework-2.png') % (directory)

        # Save chart
        fig.savefig(graph_filename)

        # Close the plot
        plt.close(fig)
