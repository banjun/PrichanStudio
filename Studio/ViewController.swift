import UIKit
import AVFoundation
import Vision

private let poseNet = PoseNet()

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var cameraSession: AVCaptureSession?
    let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInteractive)
    var previewLayer: AVCaptureVideoPreviewLayer? {
        didSet {
            oldValue?.removeFromSuperlayer()
            guard let newValue = previewLayer else { return }
            newValue.frame = view.layer.bounds
            view.layer.addSublayer(newValue)
        }
    }

    let poseNetModel = try! VNCoreMLModel(for: posenet337().model)
    private lazy var poseNetRequest: VNCoreMLRequest = {
        let req = VNCoreMLRequest(model: poseNetModel) { request, error in
            guard let obs = request.results as? [VNCoreMLFeatureValueObservation] else { return }
            let values = obs.compactMap {$0.featureValue.multiArrayValue}
            guard values.count >= 4 else { return }

            // NOTE: order of values are not ordered. labels are omitted in VNCoreMLFeatureValueObservation...
            let poses = poseNet.decodeMultiplePoses(
                scores: getTensor(values[1]),
                offsets: getTensor(values[2]),
                displacementsFwd: getTensor(values[0]),
                displacementsBwd: getTensor(values[3]),
                outputStride: 16, maxPoseDetections: 15,
                scoreThreshold: 0.5,nmsRadius: 20)

//            let output = posenet337Output(
//                heatmap__0: values[0],
//                offset_2__0: values[1],
//                displacement_fwd_2__0: values[2],
//                displacement_bwd_2__0: values[3])
            DispatchQueue.main.async {
                self.poses = poses
            }
        }
        req.imageCropAndScaleOption = .scaleFill
        return req
    }()
    var poses: [Pose] = [] {
        didSet {
            NSLog("%@", "prediction: \(String(describing: poses))")
        }
    }

    init() {
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder aDecoder: NSCoder) {fatalError()}

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        setupCameraSession()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        cameraSession?.stopRunning()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.layer.bounds
    }

    private func setupCameraSession() {
        guard cameraSession == nil else { return }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
            let input = try? AVCaptureDeviceInput(device: device) else { return }
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: videoQueue)
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA] as [String: Any]

        let session = AVCaptureSession()
        session.addInput(input)
        session.addOutput(output)
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        cameraSession = session

        session.startRunning()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let imageBuffer = sampleBuffer.imageBuffer else { return }
        let pixelBuffer = imageBuffer as CVPixelBuffer

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([poseNetRequest])
        } catch {
            NSLog("%@", "handler.perform throws: \(error)")
        }
//
//        let prediction = try? poseNet.prediction(image__0: pixelBuffer)
    }
}

