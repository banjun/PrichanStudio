import UIKit
import AVFoundation
import Vision

private let poseNet = PoseNet()
private let modelSize = CGSize(width: 337, height: 337) // see posenet337.mlmodel input
private let referenceEyeDistance: CGFloat = 320

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var cameraSession: AVCaptureSession?
    let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInteractive)
    var previewLayer: AVCaptureVideoPreviewLayer? {
        didSet {
            oldValue?.removeFromSuperlayer()
            guard let newValue = previewLayer else { return }
            newValue.frame = view.layer.bounds
            view.layer.insertSublayer(newValue, at: 0)
        }
    }

    let faceImageView = UIImageView(image: UIImage(named: "face")!)

    let poseNetModel = try! VNCoreMLModel(for: posenet337().model)
    private lazy var poseNetRequest: VNCoreMLRequest = {
        let req = VNCoreMLRequest(model: poseNetModel) { request, error in
            guard let orientation = self.cameraSession?.outputs.first?.connections.first?.videoOrientation else { return }
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
                self.posesWithOrientation = (poses, orientation)
            }
        }
        req.imageCropAndScaleOption = .scaleFill
        return req
    }()
    var posesWithOrientation: (poses: [Pose], orientation: AVCaptureVideoOrientation) = ([], .landscapeLeft) {
        didSet {
            let (poses, orientation) = posesWithOrientation
//            NSLog("%@", "prediction: \(String(describing: poses))")
            faceImageView.isHidden = poses.isEmpty
            guard let pose = poses.first else { return }
            guard let leftEye = (pose.keypoints.first {$0.part == "leftEye"}) else { return }
            guard let rightEye = (pose.keypoints.first {$0.part == "rightEye"}) else { return }
            guard let nose = (pose.keypoints.first {$0.part == "nose"}) else { return }
            let eyeDistance = CGFloat(sqrt(pow(leftEye.position.x - rightEye.position.x, 2) +
                pow(leftEye.position.y - rightEye.position.y, 2)))

            let rotation: CGFloat = {
                switch orientation {
                case .landscapeLeft: return -.pi / 2
                case .portrait: return 0
                case .landscapeRight: return .pi / 2
                case .portraitUpsideDown: return .pi
                }
            }()

            let transform = CGAffineTransform.identity
                .scaledBy(x: view.bounds.width, y: view.bounds.height)
                .translatedBy(x: +0.5, y: +0.5)
                .scaledBy(x: 1, y: -1)
                .rotated(by: rotation)
                .translatedBy(x: -0.5, y: -0.5)
                .scaledBy(x: 1 / modelSize.width, y: 1 / modelSize.height)

            let estimatedScale = (eyeDistance / min(modelSize.width, modelSize.height))
                * (min(faceImageView.intrinsicContentSize.width,
                       faceImageView.intrinsicContentSize.height) / referenceEyeDistance)

            UIView.animate(withDuration: 0.1) {
                self.faceImageView.center = CGPoint(
                    x: CGFloat(nose.position.x),
                    y: CGFloat(nose.position.y))
                    .applying(transform)
                self.faceImageView.transform = CGAffineTransform(scaleX: estimatedScale, y: estimatedScale)
            }
        }
    }

    init() {
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder aDecoder: NSCoder) {fatalError()}

    override func viewDidLoad() {
        super.viewDidLoad()

        view.addSubview(faceImageView)
    }

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
    }
}

