import * as brain from "brain.js";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { join } from "path";
const perf = require("execution-time")();

interface DataModel {
  input: string;
  output: string;
}

//DATA LATIH 102
async function latih() {
  try {
    const data: DataModel[] = JSON.parse(
      readFileSync(join(__dirname, "..", "data", "latih.json"), "utf-8")
    );
    const mData: brain.IRNNTrainingData[] = data.map((it) => it);
    const net = new brain.recurrent.LSTM();

    const exists = existsSync(join(__dirname, "..", "data", "net.json"));
    if (exists === true) {
      console.log("IMPORT NET");
      net.fromJSON(
        JSON.parse(
          readFileSync(join(__dirname, "..", "data", "net.json"), "utf-8")
        )
      );
    }

    perf.start();
    const result = net.train(mData, {
      log: true,
      logPeriod: 1000,
      iterations: 20000,
      learningRate: 0.003,
    });
    const times = perf.stop();
    console.log("ITERATIONS", result.iterations, "ERROR", result.error);
    console.log(times.time);
    writeFileSync(
      join(__dirname, "..", "data", "net.json"),
      JSON.stringify(net.toJSON())
    );
  } catch (error) {
    console.log(error);
  }
}

async function testing() {
  try {
    const data: DataModel[] = JSON.parse(
      readFileSync(join(__dirname, "..", "data", "uji.json"), "utf-8")
    );
    const net = new brain.recurrent.LSTM();
    net.fromJSON(
      JSON.parse(
        readFileSync(join(__dirname, "..", "data", "net1.json"), "utf-8")
      )
    );

    const result = data.map((it) => it.output === net.run(it.input));
    const fail = result.filter((it) => it === false).length;
    const success = result.filter((it) => it === true).length;

    console.log(
      result.length,
      "fail",
      fail,
      "success",
      success,
      "persentase",
      (success / data.length) * 100
    );

    for (const it of data) {
      // if (it.output !== net.run(it.input)) {
      //   console.log(it.input, " = ", net.run(it.input), " = ", it.output);
      // }

      console.log(
        // it.input
        // it.output
        // net.run(it.input),
        // "\t",
        net.run(it.input) === it.output ? 1 : 0
      );
    }

    // for (const iterator of data) {
    //   if (iterator.output !== net.run(iterator.input)) {
    //     console.log(iterator.output, net.run(iterator.input));
    //     trainAgain = true;
    //     // break;
    //   }
    // }
    // return trainAgain;
  } catch (error) {}
}

testing();
