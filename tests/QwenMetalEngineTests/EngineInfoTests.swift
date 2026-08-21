import XCTest
@testable import QwenMetalEngine

final class EngineInfoTests: XCTestCase {
    func testEngineInfoIdentifiesPackage() {
        XCTAssertEqual(EngineInfo.name, "qwen-metal")
        XCTAssertFalse(EngineInfo.version.isEmpty)
    }
}
