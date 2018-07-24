import UIKit

struct Heatmaps {
    var parts: [Part] // 17 elements

    struct Part {
        var data: [[Double]] // 22 x 22
        var name: String = "(part name)"
        var image: UIImage {
            let provider = CGDataProvider(dataInfo: nil, data: data.flatMap {$0.map {UInt8($0 * 255)}}, size: 22 * 22) { (_, _, _) in

                }!
            let cgImage = CGImage(width: 22,
                                  height: 22,
                                  bitsPerComponent: 8,
                                  bitsPerPixel: 8,
                                  bytesPerRow: 22,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: [],
                                  provider: provider,
                                  decode: nil,
                                  shouldInterpolate: false,
                                  intent: .defaultIntent)
            return UIImage(cgImage: cgImage!, scale: 0.25, orientation: .up)
        }
    }
}

final class HeatmapViewController: UITableViewController {
    var heatmaps: Heatmaps {
        didSet {
            DispatchQueue.main.async {
                self.tableView.reloadData()
            }
        }
    }

    init() {
        heatmaps = Heatmaps(parts: [])
        super.init(style: .plain)
    }
    required init?(coder aDecoder: NSCoder) {fatalError()}

    override func viewDidLoad() {
        super.viewDidLoad()
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "Cell")
        tableView.rowHeight = 88
        tableView.backgroundColor = UIColor(white: 1, alpha: 0.7)
    }

    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return heatmaps.parts.count
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "Cell", for: indexPath)
        let part = heatmaps.parts[indexPath.row]
        cell.backgroundColor = .clear
        cell.contentView.backgroundColor = .clear
        cell.textLabel?.text = part.name
        cell.imageView?.image = part.image
        return cell
    }

}
