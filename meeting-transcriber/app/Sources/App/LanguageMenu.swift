import SwiftUI

/// A drop-down for choosing several languages. The button names the chosen ones; the list under it has a search
/// field, the chosen languages are ticked and highlighted where they stand, and it stays open while languages are
/// picked. Plain Binding values, not @Binding: the Command Line Tools cannot expand SwiftUI's macro-backed wrappers.
struct LanguageMenu: View {
    let names: [String]                   // what can be chosen, in menu order (Languages.spokenAll / targetAll)
    let selection: Binding<[String]>
    let none: String                      // the button's text when nothing is chosen
    let isPresented: Binding<Bool>
    let query: Binding<String>
    var width: CGFloat = 260              // of the button

    var body: some View {
        Button(action: { isPresented.wrappedValue.toggle() }) {
            HStack(spacing: 6) {
                Text(Languages.summary(selection.wrappedValue, none: none)).lineLimit(1)
                Spacer(minLength: 4)
                Image(systemName: "chevron.down").font(.caption)
            }
            .frame(width: width, alignment: .leading)
        }
        .help(selection.wrappedValue.isEmpty ? none : selection.wrappedValue.joined(separator: ", "))
        .popover(isPresented: isPresented, arrowEdge: .bottom) { list }
    }

    /// The names shown: a chosen one the list does not have (typed into an older version) first, so it can be removed.
    private var shown: [String] {
        let extra = selection.wrappedValue.filter { !names.contains($0) }
        return Languages.matching(query.wrappedValue, in: extra + names)
    }

    private var list: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Search languages", text: query).textFieldStyle(.roundedBorder)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(shown, id: \.self) { name in row(name) }
                }
            }
            .frame(height: 320)
            HStack {
                Text(selection.wrappedValue.isEmpty ? "None chosen" : "\(selection.wrappedValue.count) chosen")
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Clear") { selection.wrappedValue = [] }.disabled(selection.wrappedValue.isEmpty)
                Button("Done") { isPresented.wrappedValue = false }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(10)
        .frame(width: 300)
    }

    private func row(_ name: String) -> some View {
        let chosen = selection.wrappedValue.contains(name)
        return Button(action: { toggle(name) }) {
            HStack(spacing: 8) {
                Image(systemName: chosen ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(chosen ? Color.accentColor : Color.secondary)
                Text(name).fontWeight(chosen ? .semibold : .regular)
                Spacer()
            }
            .padding(.vertical, 3)
            .padding(.horizontal, 6)
            .background(RoundedRectangle(cornerRadius: 5).fill(chosen ? Color.accentColor.opacity(0.18) : Color.clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func toggle(_ name: String) {
        var now = selection.wrappedValue
        if let index = now.firstIndex(of: name) { now.remove(at: index) } else { now.append(name) }
        selection.wrappedValue = now
    }
}
