package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const defaultRegion = "us-east-1"

var brc *bedrockruntime.Client

func init() {

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = defaultRegion
	}

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		log.Fatal(err)
	}

	brc = bedrockruntime.NewFromConfig(cfg)
}

const userMessageFormat = "[INST] %s [/INST]"
const modelIDMistralLarge = "mistral.mistral-large-2402-v1:0"
const bos = "<s>"  //beginning of string - only needed once at the start
const eos = "</s>" // end of a single conversation exchange

var verbose *bool

func main() {
	verbose = flag.Bool("verbose", false, "setting to true will log messages being exchanged with LLM")
	flag.Parse()

	reader := bufio.NewReader(os.Stdin)

	first := true
	var msg string

	for {
		fmt.Print("\nEnter your message: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if first {
			msg = bos + fmt.Sprintf(userMessageFormat, input)
		} else {
			msg = msg + fmt.Sprintf(userMessageFormat, input)
		}

		payload := MistralRequest{
			Prompt: msg,
		}

		response, err := send(payload)

		if err != nil {
			log.Fatal(err)
		}

		msg = msg + response + eos + " "

		first = false

	}
}

func send(payload MistralRequest) (string, error) {

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	if *verbose {
		fmt.Println("[request payload]", string(payloadBytes))
	}

	output, err := brc.InvokeModelWithResponseStream(context.Background(), &bedrockruntime.InvokeModelWithResponseStreamInput{
		Body:        payloadBytes,
		ModelId:     aws.String(modelIDMistralLarge),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		return "", err
	}

	fmt.Print("[Assistant]:")

	resp, err := processStreamingOutput(output, func(ctx context.Context, part []byte) error {
		fmt.Print(string(part))
		return nil
	})

	if err != nil {
		log.Fatal("streaming output processing error: ", err)
	}

	if *verbose {
		fmt.Println("[response]", resp)
	}

	return resp.Outputs[0].Text, nil
}

type MistralRequest struct {
	Prompt        string   `json:"prompt"`
	MaxTokens     int      `json:"max_tokens,omitempty"`
	Temperature   float64  `json:"temperature,omitempty"`
	TopP          float64  `json:"top_p,omitempty"`
	TopK          int      `json:"top_k,omitempty"`
	StopSequences []string `json:"stop,omitempty"`
}
type MistralResponse struct {
	Outputs []Outputs `json:"outputs"`
}
type Outputs struct {
	Text       string `json:"text"`
	StopReason string `json:"stop_reason"`
}

type StreamingOutputHandler func(ctx context.Context, part []byte) error

func processStreamingOutput(output *bedrockruntime.InvokeModelWithResponseStreamOutput, handler StreamingOutputHandler) (MistralResponse, error) {

	var combinedResult string

	resp := MistralResponse{}
	op := Outputs{}

	for event := range output.GetStream().Events() {
		switch v := event.(type) {
		case *types.ResponseStreamMemberChunk:

			var pr MistralResponse

			err := json.NewDecoder(bytes.NewReader(v.Value.Bytes)).Decode(&pr)
			if err != nil {
				return resp, err
			}

			handler(context.Background(), []byte(pr.Outputs[0].Text))

			combinedResult += pr.Outputs[0].Text
			op.StopReason = pr.Outputs[0].StopReason

		case *types.UnknownUnionMember:
			fmt.Println("unknown tag:", v.Tag)

		default:
			fmt.Println("union is nil or unknown type")
		}
	}

	op.Text = combinedResult
	resp.Outputs = []Outputs{op}

	return resp, nil
}
