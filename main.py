# main application GUI class
from App import MachineVisionApp

# region function module include
from Digital_Image_Processing_Local import Intensity_Spatial
from Digital_Image_Processing_Local import Domain_Frequency_Filtering
# endregion

# region runtime
if __name__ == '__main__':  # <-- redundant piece
    app = MachineVisionApp()
    # Intensity Transformations and Spatial Filtering (Chapter 3)
    app.transformers(Intensity_Spatial, group="Chapter 3")
    app.transformers(Domain_Frequency_Filtering, group="Chapter 4")
    app.mainloop()

# endregion