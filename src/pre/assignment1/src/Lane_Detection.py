

import config
# ****************************************************  DETECTION ****************************************************
# ****************************************************    LANES   ****************************************************

# >>>>>>>>>>>>>>>>>>>>>>>> STAGE 1 [IMPORTS] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import Segment_Colour

# >>>>>>>>>>>>>>>>>>>>>>>> STAGE 2 [IMPORTS] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import Estimate_MidLane
# >>>>>>>>>>>>>>>>>>>>>>>> STAGE 3 [IMPORTS] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import GetYellowInnerEdge
import ExtendShortLane
# >>>>>>>>>>>>>>>>>>>>>>>> STAGE 4 [IMPORTS] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import FetchInfoAndDisplay



def detect_Lane(img):
            
        # >>>>>>>>>>>>>>>>>>>>>>>> Optimization No 2 [CROPPING] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        img_cropped = img[config.CropHeight_resized:,:]

        # [Lane Detection] STAGE_1 (Segmentation) <<<<<<--->>>>>> [COLOR]:
        Mid_edge_ROI,Mid_ROI_mask,Outer_edge_ROI,OuterLane_TwoSide,OuterLane_Points = Segment_Colour(img_cropped,config.minArea_resized)

        # [Lane Detection] STAGE_2 (Estimation) <<<<<<--->>>>>> [Our Approach]:
        Estimated_midlane = Estimate_MidLane(Mid_edge_ROI,config.MaxDist_resized)

        # [Lane Detection] STAGE_3 (Cleaning) <<<<<<--->>>>>> [STEP_1]:
        OuterLane_OneSide,Outer_cnts_oneSide,Mid_cnts,Offset_correction = GetYellowInnerEdge(OuterLane_TwoSide,Estimated_midlane,OuterLane_Points)#3ms
        # [Lane Detection] STAGE_3 (Cleaning) <<<<<<--->>>>>> [STEP_2]:
        Estimated_midlane,OuterLane_OneSide = ExtendShortLane(Estimated_midlane,Mid_cnts,Outer_cnts_oneSide,OuterLane_OneSide)
        
        # [Lane Detection] STAGE_4 (Data_Extraction) <<<<<<--->>>>>> [Our Approach]:
        Distance , Curvature = FetchInfoAndDisplay(Mid_edge_ROI,Estimated_midlane,OuterLane_OneSide,img_cropped,Offset_correction)

        return Distance,Curvature
