import UIKit
import AVFoundation
import Vision

private let poseNet = PoseNet()
private let modelSize = CGSize(width: 337, height: 337) // see posenet337.mlmodel input
private let referenceEyeDistance: CGFloat = 320
private let referenceHipWidth: CGFloat = 110

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
    let skirtImageView = UIImageView(image: UIImage(named: "skirt")!)

    

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

            let heatmaps = values[1]
            self.heatmapViewController.heatmaps = Heatmaps(parts: (0..<17).map { i in
                var data: [[Double]] = []
                for y in 0..<22 {
                    var row: [Double] = []
                    for x in 0..<22 {
                        row.append(heatmaps[[NSNumber(value: i), NSNumber(value: x), NSNumber(value: y)]].doubleValue)
                    }
                    data.append(row)
                }
                return data
                }.map {Heatmaps.Part(data: $0, name: "(name)")})

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
                case .portrait: return .pi
                case .landscapeRight: return .pi / 2
                case .portraitUpsideDown: return 0
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

            // Each Hip may be in connected key points
            let adjacentKeypoints = getAdjacentKeyPoints(keypoints: pose.keypoints, minConfidence: 0.01)
            guard let leftHip = (adjacentKeypoints.flatMap {$0}.first {$0.part == "leftHip"}) else { return }
            guard let rightHip = (adjacentKeypoints.flatMap {$0}.first {$0.part == "rightHip"}) else { return }

//            NSLog("%@", "\(String(describing: (leftHip, rightHip)))")
            let hipWidth = CGFloat(sqrt(pow(leftHip.position.x - rightHip.position.x, 2) +
                pow(leftHip.position.y - rightHip.position.y, 2)))
            let estimatedSkirtScale = max(0.7, min(1.5,
                                                   hipWidth / min(modelSize.width, modelSize.height))
                * (skirtImageView.intrinsicContentSize.width / referenceHipWidth))
            let transformedLeftHipPosition = CGPoint(x: CGFloat(leftHip.position.x), y: CGFloat(leftHip.position.y)).applying(transform)
            let transformedRightHipPosition = CGPoint(x: CGFloat(rightHip.position.x), y: CGFloat(rightHip.position.y)).applying(transform)
            let angle: CGFloat = 0 //CGFloat(atan2(transformedLeftHipPosition.y - transformedRightHipPosition.y,
                          //            transformedLeftHipPosition.x - transformedRightHipPosition.x)) - .pi

            UIView.animate(withDuration: 0.1) {
                self.skirtImageView.center = CGPoint(
                    x: CGFloat(leftHip.position.x + rightHip.position.x) / 2,
                    y: CGFloat(leftHip.position.y + rightHip.position.y) / 2)
                    .applying(transform)
                self.skirtImageView.transform = CGAffineTransform(scaleX: estimatedSkirtScale, y: estimatedSkirtScale).rotated(by: angle)
            }
        }
    }

    init() {
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder aDecoder: NSCoder) {fatalError()}

    override func viewDidLoad() {
        super.viewDidLoad()

//        view.addSubview(faceImageView)
//        view.addSubview(skirtImageView)
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

    let heatmapViewController = HeatmapViewController()

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
        previewLayer?.videoGravity = .resizeAspectFill
        cameraSession = session

        session.startRunning()

        let nc = UINavigationController(rootViewController: heatmapViewController)
        heatmapViewController.preferredContentSize = CGSize(width: view.bounds.width, height: view.bounds.height / 2)
        nc.preferredContentSize = CGSize(width: view.bounds.width, height: view.bounds.height / 2)
        nc.modalPresentationStyle = .custom
        present(nc, animated: true)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let pixelBuffer = imageBuffer as CVPixelBuffer

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([poseNetRequest])
        } catch {
            NSLog("%@", "handler.perform throws: \(error)")
        }
    }
}

// copied from PoseNet-CoreML
extension ViewController {
    func getAdjacentKeyPoints(
        keypoints: [Keypoint], minConfidence: Float)-> [[Keypoint]] {

        return connectedPartIndices.filter {
            !eitherPointDoesntMeetConfidence(
                keypoints[$0.0].score,
                keypoints[$0.1].score,
                minConfidence)
            }.map { [keypoints[$0.0],keypoints[$0.1]] }
    }

    func eitherPointDoesntMeetConfidence(
        _ a: Float,_ b: Float,_ minConfidence: Float) -> Bool {
        return (a < minConfidence || b < minConfidence)
    }
}

