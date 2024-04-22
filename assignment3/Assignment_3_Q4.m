function main()
    % Read the video file
    vidFile = 'CV Video.mp4';
    vidReader = VideoReader(vidFile);

    % Parameters for reference frames
    refFrames = [1, 11, 31];

    % Optical flow parameters
    opticFlow = opticalFlowFarneback('NumPyramidLevels',3, 'PyramidScale',0.5, 'NumIterations',15, 'NeighborhoodSize',7, 'FilterSize',5);

    % Define the output video
    outVideo = VideoWriter('output.mp4', 'MPEG-4');
    outVideo.FrameRate = vidReader.FrameRate;
    open(outVideo);

    % Read and process each frame
    processFrames(vidReader, refFrames, opticFlow, outVideo);

    % Close the video writer
    close(outVideo);
end

function processFrames(vidReader, refFrames, opticFlow, outVideo)
    % Read the first frame
    prevFrame = readFrame(vidReader);
    prevGray = rgb2gray(prevFrame);

    % Process each frame
    while hasFrame(vidReader)
        frame = readFrame(vidReader);
        gray = rgb2gray(frame);
        
        % Loop over the reference frames
        for i = 1:length(refFrames)
            if refFrames(i) == 1 || mod(vidReader.CurrentTime*vidReader.FrameRate, refFrames(i)) == 0
                % Calculate optical flow
                flow = estimateFlow(opticFlow, prevGray);
                
                % Plot optical flow vectors
                frameWithFlow = plotOpticalFlow(frame, flow);
                
                % Write frame with optical flow to video
                writeVideo(outVideo, frameWithFlow);
            end
        end
        
        % Update previous frame
        prevGray = gray;
    end
end

function frameWithFlow = plotOpticalFlow(frame, flow)
    % Plot optical flow vectors
    imshow(frame);
    hold on;
    plot(flow, 'DecimationFactor', [10 10], 'ScaleFactor', 2);
    hold off;
    
    % Convert figure to frame
    drawnow;
    frameWithFlow = getframe;
    
    % Resize frame to match original frame size
    frameWithFlow = imresize(frameWithFlow.cdata, [size(frame, 1), size(frame, 2)]);
end
